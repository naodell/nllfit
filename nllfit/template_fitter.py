'''
    Class for template fitting with nll estimation.  
    
    TODO:
    =====
    * add chi^2 minimization
    * handle higher dimensional distributions
    * fit multiple distributions simultaneously
    * functionality for using unbinned data and non-binned templates
'''

from functools import partial

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, beta
import matplotlib as mpl
import matplotlib.pyplot as plt

import numdifftools as nd

mpl.style.use('default')
params = {'legend.fontsize': 20,
          'axes.labelsize': 20,
          'axes.titlesize':'x-large',
          'xtick.labelsize':18,
          'ytick.labelsize':18,
         }
mpl.rcParams.update(params)

class TemplateFitter():
    '''
    Fits templates to provided dataset.  Currently only works for binned data.
    '''
    def __init__(self, templates, bins=None, params_init=None):
        self._templates = self._initialize_templates(templates)
        self._ntemp     = np.shape(templates)[0]
        self._bins      = bins
        self._params    = np.array((self._ntemp - 1)*[1./np.size(templates), ]) if params_init==None else params_init

    def _initialize_templates(self, templates):
        '''
        Normalize binned templates to unity.
        '''
        normed_templates = [] 
        for t in templates:
            normed_templates.append(t/np.sum(t))
        return normed_templates

    def calc_nll(self, params, data, templates):
        '''
        Calculates the nll given the (digitized) data.
        '''
        a = np.concatenate((params, [1 - np.sum(params)]))
        p = [t[data] for t in templates]
        f = np.dot(a, p)
        cost = -np.sum(np.log(f))
        return cost

    def fit(self, data):
        '''
        Fit to the provided dataset
        '''
        digi_data = np.digitize(data, self._bins[:-1]) - 1
        result = minimize(self.calc_nll, self._params,
                          method      = 'SLSQP',
                          bounds      = (self._ntemp - 1)*[(0,1),],
                          args        = (digi_data, self._templates),
                          #constraints = {'eq':lambda a: 1 - np.sum(a)}
                         )
        return result

    def scan_nll(self, scan_vals, data, param_min):
        '''
        Scans the nll for each mixture parameter while holding the remaining parameters fixed.
        '''
        digi_data = np.digitize(data, self._bins[:-1]) - 1
        func      = partial(self.calc_nll, data=digi_data, templates=self._templates)
        nll_scan  = np.array([func(x) for x in scan_vals]) - func(param_min)
        ddfunc    = nd.Derivative(func, n=2)
        sig       = np.sqrt(1/ddfunc(param_min))
        
        return nll_scan, sig

### Some plotting functions ###

def plot_result():
    '''
    helper function for overlaying prefit and post templates.
    '''

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), facecolor='white', sharex=True, gridspec_kw={'height_ratios':[3,1]})
    fig.subplots_adjust(hspace=0)
    x, y, yerr = pt.hist_to_errorbar(df_data['lepton1_pt'], hist_lut.n_bins, (hist_lut.xmin, hist_lut.xmax))
    axes[0].errorbar(x, y, yerr,
                     fmt        = 'ko',
                     capsize    = 0,
                     elinewidth = 2
                    )

    hdata  = [df['lepton1_pt'].values for df in df_split]
    h_pre, b, _ = axes[0].hist(hdata, 
                               bins=hist_lut.n_bins, 
                               range=(hist_lut.xmin, hist_lut.xmax), 
                               linewidth=2,
                               color=['C9', 'C0'], 
                               histtype='step', 
                               stacked=True, 
                               weights=[df['weight'] for df in df_split]
                              )

    # calculate default mixture and error
    num = h_pre[0].sum()
    den = h_pre[-1].sum()
    alpha0 = num/den
    alpha0_err = [np.abs(alpha0 - beta.ppf(0.16, num, den - num + 1)), 
                  np.abs(alpha0 - beta.ppf(0.84, num + 1, den - num))]
    h_post, b, _ = axes[0].hist(hdata, 
                                bins=hist_lut.n_bins, 
                                range=(hist_lut.xmin, hist_lut.xmax), 
                                linewidth=2,
                                color=['C3', 'C1'], 
                                histtype='step', 
                                linestyle='dashed',
                                stacked=True, 
                                weights=[(result.x[0]/alpha0)*df_split[0]['weight'], (1 - result.x[0])/(1 - alpha0)*df_split[1]['weight']]
                               )
    ## calculate ratios
    herr = np.sqrt(mc_hists['lepton1_pt'][0][1] + mc_hists['lepton1_pt'][1][1]) 
    ratio_pre = y/h_pre[-1]
    ratio_err = np.sqrt(y/h_pre[-1]**2 + (y*herr/h_pre[-1]**2)**2)
    axes[1].errorbar(x, ratio_pre, ratio_err,
                     fmt        = 'C0o',
                     ecolor     = 'C0',
                     capsize    = 0,
                     elinewidth = 3,
                     alpha = 1.
                    )

    ratio_post = y/h_post[-1]
    ratio_err = np.sqrt(y/h_post[-1]**2 + (y*herr/h_post[-1]**2)**2)
    axes[1].errorbar(x, ratio_post, ratio_err,
                     fmt        = 'C1o',
                     ecolor     = 'C1',
                     capsize    = 0,
                     elinewidth = 3,
                     alpha = 1.
                    )

    axes[1].grid()
    axes[1].set_xlabel(r'$\sf p_{T}^{\mu}$ [GeV]')
    axes[1].set_ylabel('Data / MC')
    axes[1].set_ylim(0.5, 1.5)
    axes[1].legend(['prefit', 'postfit'], loc=1, fontsize=16)
    axes[1].plot([hist_lut.xmin, hist_lut.xmax], [1, 1], 'r--')

    axes[0].grid()
    axes[0].set_ylabel(r'Entries / 5 GeV')
    axes[0].set_xlim(10, 150)
    axes[0].legend([r'$W\rightarrow \mu$ (prefit)', r'$W\rightarrow\tau\rightarrow \mu$ (prefit)', 
                    r'$W\rightarrow \mu$ (postfit)', r'$W\rightarrow\tau\rightarrow \mu$ (postfit)', 'data'])

    axes[0].text(85, 1800, r'$\alpha_{postfit} = $' + f' {result.x[0]:3.4} +/- {sig:2.2}', {'size':18})
    axes[0].text(85, 1600, r'$\alpha_{prefit} = $' + f' {alpha0:3.4} +/- {np.mean(alpha0_err):2.2}', {'size':18})

    plt.savefig('plots/fit_mu_channel.png')
    plt.show()
