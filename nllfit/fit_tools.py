#!/usr/bin/env python

from __future__ import division

import pandas as pd
import numpy as np
import numpy.random as rng
from numpy.polynomial.legendre import legval
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import chi2, norm
from scipy.special import wofz, erf

from numba import jit

# global options
np.set_printoptions(precision=3.)


# Data manipulation #
def scale_data(x, xmin=12., xmax=70., invert=False):

    if not invert:
        return 2*(x - xmin)/(xmax - xmin) - 1
    else:
        return 0.5*(x + 1)*(xmax - xmin) + xmin


def get_data(filename, varname):
    '''
    Get data from file and convert to lie in the range [-1, 1]
    '''
    ntuple  = pd.read_csv(filename)
    data    = ntuple[varname].values
    n_total = data.size

    return data, n_total


# PDF definitions (maybe put these in a separate file)
@jit
def lorentzian(x, a):
    '''
    Lorentzian line shape

    Parameters:
    ===========
    x: data
    a: model parameters (mean and HWHM)
    '''
    return a[1]/(np.pi*((x-a[0])**2 + a[1]**2))


@jit
def voigt(x, a):
    '''
    Voigt profile

    Parameters:
    ===========
    x: data
    a: model paramters (mean, gamma, and sigma)
    '''
    mu    = a[0]
    gamma = a[1]/2
    sigma = a[2]

    if gamma == 0:
        return norm.pdf(x, [mu, sigma])
    elif sigma == 0:
        return lorentzian(x, [mu, gamma])
    else:
        z = ((x - mu) + 1j*gamma)/(sigma*np.sqrt(2))
        y = np.real(wofz(z))/(sigma*np.sqrt(2*np.pi))
        return y


@jit
def crystal_ball(x, a):
    '''
    Crystal ball is a power law stitched together with a Gaussian that accounts
    for lossy measurements.

    Parameters:
    ===========
    x: data
    a: model parameters (mean, alpha, sigma, n)
    '''

    x0    = a[0]
    sigma = a[1]
    n     = a[2]
    alpha = np.abs(a[3])

    z = (x - x0)/sigma
    A = np.power(n/alpha, n)*np.exp(-alpha**2/2.)
    B = n/alpha - alpha
    C = n/(alpha*(n - 1.))*np.exp(-alpha**2/2.)
    D = np.sqrt(np.pi/2.)*(1. + erf(alpha/np.sqrt(2)))
    N = 1./(sigma*(C + D))

    if type(z) == np.ndarray:
        zplus, zminus = z[z > -alpha], z[z <= -alpha]
        z[z > -alpha], z[z <= -alpha]  = N*np.exp(-zplus**2/2.), N*A*np.power(B - zminus, -n)
        return z
    else:
        if z > -alpha:
            return N*np.exp(-z**2/2.)
        else:
            return N*A*np.power(B - z, -n)


@jit
def legendre(x, a, xlim=(-1, 1)):
    '''
    Nth order Legendre Polynomial with constant term set to 0.5 to enforce
    normalization.  

    Parameters:
    ===========
    x: data
    a: an array of coefficients for the polynomial terms.  The order of the
       polynomial will be equal to the length of a
    '''
    p = np.concatenate(([0.5], a))
    z = scale_data(x, xmin=xlim[0], xmax=xlim[1])
    f = legval(z, p)*2./(xlim[1] - xlim[0])

    return f

@jit
def smeared_exp(x, tau, sigma):
    '''
    An exponential convolved with a Gaussian.

    Parameters:
    ===========
    x: data
    tau: lifetime of exponential
    sigma: width of Gaussian convolution kernel
    '''
    f  = 1./(2*tau) * np.exp(-x/tau + sigma**2/(2*tau**2))
    f *= (1 - erf((sigma**2/tau - x)/(sigma*np.sqrt(2))))

    return f

@jit
def double_exp(x, a):
    '''
    Double sided exponentional distribution.

    Parameters:
    ===========
    x: data
    tau: inverse decay constant
    '''

    tau = a[0]
    if type(x) == np.ndarray:
        x[x > 0.], x[x <= 0.]  = tau*np.exp(-x/tau), tau*np.exp(x/tau)
        return x
    else:
        if x > 0.:
            return tau*np.exp(-x/tau)
        else:
            return tau*np.exp(x/tau)

# toy MC p-value calculator #
def calc_local_pvalue(N_bg, var_bg, N_sig, var_sig, ntoys=1e7):

    print('')
    print('Calculating local p-value and significance based on {0} toys...'.format(int(ntoys)))
    toys    = rng.normal(N_bg, var_bg, int(ntoys))
    pvars   = rng.poisson(toys)
    pval    = pvars[pvars > N_bg + N_sig].size/pvars.size
    print('local p-value = {0}'.format(pval))
    print('local significance = {0:.2f}'.format(np.abs(norm.ppf(pval))))

    return pval


# Monte Carlo simulations #
def lnprob(x, pdf, bounds):
    if np.any(x < bounds[0]) or np.any(x > bounds[1]):
        return -np.inf
    else:
        return np.log(pdf(x))


def generator(pdf, bounds, ntoys):
    '''
    Rejection sampling with broadcasting gives approximately the requested
    number of toys.  This works okay for simple pdfs.

    Parameters:
    ===========
    pdf             : the pdf that will be sampled to produce the synthetic data
    bounds          : specify (lower, upper) bounds for the toy data
    ntoys           : number of synthetic datasets to be produced
    '''

    # Generate random numbers and map into domain defined by bounds.  Generate
    # twice the number of requested events in expectation of ~50% efficiency.
    # This will not be the case for more peaked pdfs
    rnums = rng.rand(2, int(3*ntoys))
    x = rnums[0]
    x = (bounds[1] - bounds[0])*x + bounds[0]
    y = pdf(x)

    # Carry out rejection sampling
    rnums[1] = rnums[1]*(y.max()/rnums[1].max())
    mask = y > rnums[1]
    x    = x[mask]

    # if the exact number of toy datasets are not generated either trim or
    # produce more.
    if x.size < ntoys:
        x_new = generator(pdf, bounds, ntoys-x.size)
        x = np.concatenate((x, x_new))
    else:
        x = x[:int(ntoys)]

    return x


######################
### plotting tools ###
######################

def plot_pvalue_scan_1D(qscan, x, path):
    '''
    Helper function for plotting 1D pvalue scans.
    '''

    p_val = np.array(0.5*chi2.sf(qscan, 1) + 0.25*chi2.sf(qscan, 2))
    plt.plot(x, p_val)

    # Draw significance lines
    ones = np.ones(x.size)
    plt.plot(x, norm.sf(1)*ones, 'r--')
    for i in xrange(2, 7):
        if norm.sf(i) < p_val.min:
            break
        plt.plot(x, norm.sf(i)*ones, 'r--')
        plt.text(60, norm.sf(i)*1.25, r'${0} \sigma$'.format(i), color='red')

    plt.yscale('log')
    plt.title(r'')
    plt.ylim([np.min(0.5*np.min(p_val), 0.5*norm.sf(3)), 1.])
    plt.xlim([x[0], x[-1]])
    plt.xlabel(r'$m_{\mu\mu}$ [GeV]')
    plt.ylabel(r'$p_{local}$')
    plt.savefig(path)
    plt.close()

def plot_pvalue_scan_2D(qscan, x, y, path, nchannels=1):
    '''
    Helper function for plotting 1D pvalue scans.
    '''
    if nchannels == 1:
        p_val = np.array(0.5*chi2.sf(qscan, 1))
    elif nchannels == 2:
        p_val = np.array(0.5*chi2.sf(qscan, 1) + 0.25*chi2.sf(qscan, 2))

    p_val = p_val.reshape(x.size, y.size).transpose()
    z_val = -norm.ppf(p_val)

    ### draw the p values as a colormesh
    plt.pcolormesh(x, y, p_val[:-1, :-1],
                   cmap='viridis_r',
                   norm=LogNorm(vmin=0.25*p_val.min(),
                                vmax=p_val.max()),
                   linewidth=0,
                   rasterized=True)
    cbar = plt.colorbar()
    cbar.set_label(r'$p_{local}$')

    # draw the z scores as contours
    vmap = plt.get_cmap('gray_r')
    vcol = [vmap(0.95) if i >= 2 else vmap(0.05) for i in range(5)]
    cs = plt.contour(x, y, z_val, [1, 2, 3, 4, 5], colors=vcol)
    plt.clabel(cs, inline=1, fontsize=10, fmt='%d')

    plt.xlabel(r'$m_{\mu\mu}$ [GeV]')
    plt.ylabel(r'$\sigma$ [GeV]')
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.savefig(path)
    plt.close()


def fit_plot_1D(data, sig_model, bg_model, xlim, nbins=20, suffix='mumu', path=None):

    # Scale pdfs and data from [-1, 1] back to the original values
    N       = data.size
    binning = (xlim[1] - xlim[0])/nbins

    params  = sig_model.get_parameters()
    x       = np.linspace(xlim[0], xlim[1], num=10000)
    y_sig   = N*binning*sig_model.pdf(x)
    y_bg1   = (1 - params['A'])*N*binning*bg_model.pdf(x, params)
    y_bg2   = N*binning*bg_model.pdf(x)

    # Get histogram of data points
    h = plt.hist(data, bins=nbins, range=xlim)
    bincenters  = (h[1][1:] + h[1][:-1])/2.
    binerrs     = np.sqrt(h[0])
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    ax.plot(x, y_sig, 'b-', linewidth=2.5)
    ax.plot(x, y_bg1, 'b--', linewidth=2.5)
    ax.plot(x, y_bg2, 'r-.', linewidth=2.5)
    ax.errorbar(bincenters, h[0], yerr=binerrs,
                fmt='ko', capsize=0, elinewidth=2, markersize=5)
    ax.legend(['BG+Sig.', 'BG', 'BG only', 'Data'], loc=1)

    if suffix == 'mumu':
        ax.set_xlabel(r'$\sf m_{\mu\mu}$ [GeV]')
    elif suffix == 'emu':
        ax.set_xlabel(r'$\sf m_{e\mu}$ [GeV]')
    elif suffix == 'ee':
        ax.set_xlabel(r'$\sf m_{ee}$ [GeV]')
    elif suffix == 'hgg':
        ax.set_title(r'$\sf h_{0}(125)\rightarrow \gamma\gamma$')
        ax.set_xlabel(r'$\sf m_{\gamma\gamma}$ [GeV]')
    elif suffix == 'hzz':
        ax.set_title(r'$\sf h_{0}(125)\rightarrow 4\ell$')
        ax.set_xlabel(r'$\sf m_{4\ell}$ [GeV]')

    else:
        ax.set_xlabel('x')

    ax.set_ylabel('Entries / 2 GeV')
    ax.set_ylim([0., 1.4*np.max(h[0])])
    ax.set_xlim(xlim)
    ax.grid()

    ### Add lumi text ###
    #ax.text(0.06, 0.9, r'$\bf CMS$', fontsize=30, transform=ax.transAxes)
    #ax.text(0.17, 0.9, r'$\it Preliminary $', fontsize=20, transform=ax.transAxes)
    #ax.text(0.68, 1.01, r'$\sf{19.7\,fb^{-1}}\,(\sqrt{\it{s}}=8\,\sf{TeV})$', fontsize=20, transform=ax.transAxes)

    if path is not None:
        fig.savefig('{0}/mass_fit_{1[1]}_{1[2]}.pdf'.format(path, suffix))
        fig.savefig('{0}/mass_fit_{1[1]}_{1[2]}.png'.format(path, suffix))
        plt.close()

def ks_test(data, model_pdf, xlim=(-1, 1), make_plots=False, suffix=None):
    '''
    Kolmogorov-Smirnov test.  Returns the residuals of |CDF_model - CDF_data|.
    '''

    n_points = 1e5
    x = np.linspace(xlim[0], xlim[1], n_points)
    pdf = model_pdf(x)
    cdf = np.cumsum(pdf)*(xlim[1] - xlim[0])/n_points

    data.sort()
    x_i = np.array([np.abs(d - x).argmin() for d in data])
    cdf_i = np.linspace(1, data.size, data.size)/data.size

    ks_residuals = np.abs(cdf[x_i] - cdf_i)

    if make_plots:
        plt.hist(ks_residuals, bins=25, histtype='step')
        plt.ylabel('Entries')
        plt.xlabel(r'$|\rm CDF_{model} - CDF_{data}|$')
        plt.savefig('plots/ks_residuals_{0}.pdf'.format(suffix))
        plt.close()

        plt.plot(x, cdf)
        plt.plot(data, cdf_i)
        plt.ylabel('CDF(x)')
        plt.xlabel('x')
        plt.title(suffix)
        plt.legend(['model', 'data'])
        plt.savefig('plots/ks_cdf_overlay_{0}.pdf'.format(suffix))
        plt.close()

    return ks_residuals
