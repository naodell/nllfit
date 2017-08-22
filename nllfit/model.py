from __future__ import division

import numpy as np
from lmfit import Parameters

class Model:
    '''
    Model class that will be passed to NLLFitter to fit to a dataset.  Requires
    that the model pdf be specified and the model parameters.  The model
    parameters should be provided in a lmfit Parameters class object

    Parameters
    ==========
    pdf        : a function describing the model that takes argurements (params, data)
    parameters : lmfit Parameter object
    '''
    def __init__(self, pdf, parameters):
        self._pdf        = pdf
        self._parameters = parameters
        self.corr        = None

    def get_parameter(self, name, by_value=True):
        '''
        Returns parameter parameter value or lmfit Parameter object

        Parameters:
        ===========
        name    : key for paramter value
        by_value: if True returns a list of parameter values, otherwise the
                  function will return the lmfit Parameters object
        '''
        if by_value:
            return [p.value for p in self._parameters.values()]
        else:
            return self._parameters

    def get_parameters(self, by_value=False):
        '''
        Returns parameters either as an lmfit Parameters object or a list of parameter values

        Parameters:
        ===========
        by_value: if True returns a list of parameter values, otherwise the
                  function will return the lmfit Parameters object
        '''
        if by_value:
            return [p.value for p in self._parameters.values()]
        else:
            return self._parameters

    def fix_parameters(self, names=None):
        '''
        Fixes parameters to their current value.  If names are not specified,
        all parameters are fixed.

        Parameters:
        ===========
        names: paramters to fix. 
        '''
        
        for n, _ in self._parameters.items():
            if names != None:
                if n in names:
                    self._parameters[n].vary = False
            else:
                self._parameters[n].vary = False

    def get_bounds(self):
        '''
        Return list of tuples with the bounds for each parameter.
        '''
        return [(p.min, p.max) for n, p in self._parameters.items()]

    def set_bounds(self, param_name, xmin, xmax):
        '''
        Set lower and upper bounds for parameter with the given name.
        '''
        self._parameters[param_name].min = xmin
        self._parameters[param_name].max = xmax

    def get_constranints(self):
        '''
        Return list of tuples with the bounds for each parameter.
        '''
        return [p.expr for n, p in self._parameters.items()]

    def pdf(self, data, params=None):
        '''
        Returns the pdf as a function with current values of parameters
        '''
        if isinstance(params, Parameters):
            return self._pdf(data, [params[n].value for n in self._parameters.keys()])
        if isinstance(params, np.ndarray):
            return self._pdf(data, params)
        else:
            return self._pdf(data, self.get_parameters(by_value=True))

    def set_parameter_value(self, param_name, value):
        '''
        Change the value of named parameter.
        '''
        self._parameters[param_name].value = value

    def update_parameters(self, params, covariance=None):
        '''
        Updates the parameters values and errors of each of the parameters if
        specified.  Parameters can be specified either as an lmfit Parameters
        object or an array.

        Parameters:
        ===========
        params: new values of parameters
        covariance: result from _get_corr(), i.e., the uncertainty on the
                    parameters and their correlations in a tuple (sigma, correlation_matrix)
        '''

        for i, (pname, pobj) in enumerate(self._parameters.items()):
            if isinstance(params, np.ndarray):
                self._parameters[pname].value = params[i]
            else:
                self._parameters[pname] = params[pname]

            if covariance:
                self._parameters[pname].stderr = covariance[0][i]

        if covariance:
            self.corr = covariance[1]

    def calc_nll(self, params, data):
        '''
        Return the negative log likelihood of the model given some data.

        Parameters
        ==========
        a: model parameters specified as a numpy array or lmfit Parameters
           object.  If not specified, the current model parameters will be used
        data: data points where the PDF will be evaluated
        '''
        if np.any(params) is None:
            params = [p.value for p in self._parameters.values()]
        elif isinstance(params, Parameters):
            params = [params[k].value for k in self._parameters.keys()]

        pdf = self._pdf(data, params)
        # remove underflow, overflow, and nan
        pdf = pdf[(pdf != np.nan) & (pdf != np.inf) & (pdf != 0)]
        # scale nll based on size of input dataset to keep value from getting too large
        nll = -np.sum(np.log(pdf))

        return nll


class CombinedModel(Model):
    '''
    Combines multiple models so that their PDFs can be estimated
    simultaneously.

    Parameters
    ==========
    models: an array of Model instances
    '''
    def __init__(self, models):
        self.models = models
        self._initialize()
        self.corr = None

    def _initialize(self):
        '''
        Returns a dictionary of parameters where the keys are the parameter
        names and values are tuples with the first entry being the parameter
        value and the second being the uncertainty on the parameter.
        '''
        params = Parameters()
        nparams = []  # keep track of how many parameters each model has
        for m in self.models:
            p = m.get_parameters()
            nparams.append(len(p))
            params += p
        self._parameters = params
        self._nparams = nparams

    def calc_nll(self, params, data):
        '''
        Wrapper for Model.calc_nll.  Converts params to an lmfit Parameter
        object which can then be unpacked by Model.calc_nll.  (Probably not the
        most efficient thing to do, but it makes dealing with sorting which
        parameters got with what model.)

        Parameters"
        ===========
        params: parameters of the combined model
        data: an array of datasets; one dataset per model
        '''

        if len(data) is not len(self.models):
            print('The number of datasets must be the same as the number of models!!!')
            print('There are {0} models'.format(len(self.models)))
            return

        if isinstance(params, np.ndarray):
            self.update_parameters(params)

        params = self.get_parameters()

        nll         = 0.
        param_count = 0
        for i, (m, x) in enumerate(zip(self.models, data)):
            nll         += m.calc_nll(x, params)
            param_count += self._nparams[i]

        return nll
