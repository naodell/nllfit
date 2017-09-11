'''
nll fitter base class
'''

from itertools import product

import numpy as np
import numdifftools as nd
from scipy.optimize import minimize, basinhopping
from lmfit import report_fit
from functools import partial


class NLLFitter:
    '''
    Class for estimating PDFs using negative log likelihood minimization.  Fits
    a Model class to a dataset.

    Parameters:
    ==========
    model    : a Model object or and array of Model objects
    data     : the dataset or datasets we wish to carry out the modelling on
    min_algo : algorith used for minimizing the nll (uses available scipy.optimize algorithms)
    verbose  : control verbosity of fit method
    '''
    def __init__(self, model, min_algo='SLSQP', verbose=True):
        self._model    = model
        self.min_algo  = min_algo
        self.verbose   = verbose
        self.aux_cost  = None

    def _objective(self, params, data):
        '''
        Default objective function.  Perhaps it would make sense to make this
        easy to specify.  If an auxiliary cost function is included it will be
        accounted for here by being added to the nll cost.  

        Parameters:
        ==========
        params: model parameters in an numpy array
        data: dataset to calculate the NLL on
        '''

        model_params = self._model.get_parameters()
        params       = np.array([params[i] if p.vary else p.value
                                 for i, p in enumerate(model_params.values())])
        obj = 0.

        nll = self._model.calc_nll(params, data)
        if nll is not np.nan:
            obj += nll

        return obj

    def _get_corr_mc(self, x, params, ntoys=100):
        '''
        Calculate covariance using MC 

        Parameters:
        ===========
        x      : the data
        params : parameter values at which the Hessian will be evaluated.
        '''
        pass

    def _get_corr(self, x, params):
        '''
        Calculates covariance matrix for model parameters by calculating the
        Hessian of the NLL conditioned on the dataset being fit to.

        Parameters:
        ===========
        x      : the data
        params : parameter values at which the Hessian will be evaluated.
        '''
        f_obj = partial(self._model.calc_nll, data=x)
        hcalc = nd.Hessian(f_obj, 
                           step=1e-2, #[5e-3*p for p in params], 
                           method='central', 
                           full_output=True
                          )

        hobj = hcalc(params)[0]
        if np.linalg.det(hobj) != 0:
            hinv        = np.linalg.pinv(hobj)
            # calculate the full covariance matrix in the case that the Hessian is non-singular
            sig         = np.sqrt(hinv.diagonal())
            corr_matrix = hinv/np.outer(sig, sig)
            return sig, corr_matrix

        else:
            print('Hessian matrix is singular! Cannot calculate covariance matrix of the likelihood')
            # if the Hessian is singular, try to just get its diagonal for parameter errors
            sig         = params 
            corr_matrix = np.identity(np.size(params))
            return sig, corr_matrix


    def fit(self, data, params_init=None, calculate_corr=True, mode='local'):
        '''
        Fits the model to the given dataset using scipy.optimize.minimize.
        Returns the fit result object.

        Parameter:
        ==========
        data           : dataset to be fit the model to
        params_init    : initialization parameters; if not specified, current values are used
        calculate_corr : specify whether the covariance matrix should be
                         calculated.  If true, this will do a numerical calculation of the
                         covariance matrix based on the currenct objective function about the
                         minimum determined from the fit
        mode           : determines whether optimize.minize ('local') or optimize.basinhopping ('global') is used
        '''

        if params_init:
            self._model.update_params(params_init)
        else:
            params_init = self._model.get_parameters(by_value=True)

        if mode == 'local':
            result = minimize(self._objective, params_init,
                              method = self.min_algo,
                              bounds = self._model.get_bounds(),
                              #constraints = self._model.get_constraints(),
                              args   = (data),
                              options = {
                                  'ftol':1e-6*np.sqrt(data.size),
                                  'eps':1.5e-8*np.sqrt(data.size),
                                  }
                              )
        elif mode == 'global':
            result = basinhopping(self._objective,
                                  params_init,
                                  niter=200,
                                  minimizer_kwargs = { 
                                      'bounds':self._model.get_bounds(),
                                      'method':self.min_algo, 
                                      'args':(data),
                                      #'constraints':self._model.get_constraints(),
                                      'options':{
                                          'ftol':1e-6*np.sqrt(data.size),
                                          'eps':1.5e-8*np.sqrt(data.size),
                                          }
                                      }
                                 )
            if result.fun != np.nan:
                result.status = 0
            else:
                result.status = -1

        if self.verbose:
            print('Fit finished with status: {0}\n'.format(result.status))

        if result.status == 0:
            if calculate_corr:
                sigma, corr = self._get_corr(data, result.x)
                #sigma, corr = self._get_corr_mc(data, result.x)
            else:
                sigma, corr = result.x, 0.

            self._model.update_parameters(result.x, (sigma, corr))
            if self.verbose:
                report_fit(self._model.get_parameters(), show_correl=False)
                #report_fit(result, show_correl=False)
                print('')
                print('[[Correlation matrix]]')
                print(corr, '\n')

        return result

    def scan(self, scan_params, data, amps=None):
        '''
        Fits model to data while scanning over give parameters.

        Parameters:
        ===========
        scan_params : ScanParameters class object specifying parameters to be scanned over
        data        : dataset to fit the models to
        amps        : indices of signal amplitude parameters
        '''

        ### Save bounds for parameters to be scanned so that they can be reset
        ### when finished
        saved_bounds = {}
        params = self._model.get_parameters()
        for name in scan_params.names:
            saved_bounds[name] = (params[name].min, params[name].max)

        nllscan     = []
        dofs        = []  # The d.o.f. of the field will vary depending on the amplitudes
        best_params = 0.
        nll_min     = 1e9
        scan_vals, scan_div = scan_params.get_scan_vals()
        for i, scan in enumerate(scan_vals):
            ### set bounds of model parameters being scanned over
            for j, name in enumerate(scan_params.names):
                self._model.set_bounds(name, scan[j], scan[j]+scan_div[j])
                self._model.set_parameter_value(name, scan[j])

            ### Get initialization values
            params_init = [p.value for p in params.values()]
            result = minimize(self._objective,
                              params_init,
                              method = self.min_algo,
                              bounds = self._model.get_bounds(),
                              #constraints = self._model.get_constraints(),
                              args   = (data)
                              )

            if result.status in (0, 8):
                nll = self._model.calc_nll(result.x, data)
                nllscan.append(nll)
                if nll < nll_min:
                    best_params = result.x
                    nll_min = nll

                if amps:
                    dofs.append(np.sum(result.x[amps] > 1e-6))
            else:
                continue

        ## Reset parameter bounds
        for name in scan_params.names:
            self._model.set_bounds(name, saved_bounds[name][0], saved_bounds[name][1])

        nllscan = np.array(nllscan)
        dofs = np.array(dofs)
        return nllscan, best_params, dofs

class ScanParameters:
    '''
    Class for defining parameters for scanning over fit parameters.
    Parameters
    ==========
    names: name of parameters to scan over
    bounds: values to scan between (should be an array with 2 values)
    nscans: number of scan points to consider
    '''
    def __init__(self, names, bounds, nscans, fixed=False):
        self.names  = names
        self.bounds = bounds
        self.nscans = nscans
        self.init_scan_params()

    def init_scan_params(self):
        scans = []
        div   = []
        for n, b, ns in zip(self.names, self.bounds, self.nscans):
            scans.append(np.linspace(b[0], b[1], ns))
            div.append(np.abs(b[1] - b[0])/ns)
        self.div   = div
        self.scans = scans

    def get_scan_vals(self, ):
        '''
        Return an array of tuples to be scanned over.
        '''
        scan_vals = list(product(*self.scans))
        return scan_vals, self.div
