import numpy as np


import sys
import os
import traceback
import optparse
import time
import logging


def get_nQ(T_eV, m=9.10938215e-28):
    h = 6.62607004e-27 # erg-s
    hbar = h / 2 / np.pi # erg-s = g cm2 / s
    e_mks = 1.6021766208e-19 # C, J / eV
    erg_per_eV = e_mks * 1e7 # erg / eV
    nQ = ( m * erg_per_eV * T_eV / (2 * np.pi * hbar**2) )**(3/2)# cm-3
    return nQ # cm-3



def sackurTetrode(n, m, V_cm3, T_eV):
    kB = 1.38064852e-16 # erg / K
    N = n * V_cm3 #
    nQ = get_nQ(T_eV, m) # cm-3
    S = N * kB * (5/2 + np.log(nQ / n)) # erg / K
    return S 


def FD_int_num( eta, j, tol=1e-12, Nmax=600, fType='roman' ):
    r"""FD_int_num.m”
    function [ y N err ] = FD_int_num( eta, j, tol, Nmax )
    Numerical integration of Fermi-Dirac integrals for order j > -1.
    Author: Raseong Kim (Purdue University)
    Date: September 29, 208
    Extended (composite) trapezoidal quadrature rule with variable
    transformation, x = exp( t - exp( t ) )
    Valid for eta ~< 15 with precision ~eps with 60~500 evaluations. #
    Inputs
    eta: eta_F
    j: FD integral order
    tol: tolerance
    Nmax: number of iterations limit #
    Note: When "eta" is an array, this function should be executed
    repeatedly for each component. #
    Outputs
    y: value of FD integral (the "script F" defined by Blakemore (1982))
    N: number of iterations
    err: error #
    For more information in Fermi-Dirac integrals, see:
    "Notes on Fermi-Dirac Integrals (3rd Edition)" by Raseong Kim and Mark
    Lundstrom at http://nanohub.org/resources/5475 #
    Reference
    [1] W. H. Press, S. A. Teukolsky, W. T. Vetterling, and B. P. Flannery,
    Numerical recipies: The art of scientific computing, 3rd Ed., Cambridge
    University Press, 2007.
    """
    import numpy as np
    import scipy.special as spec
    
    if eta < -20:
        y = np.exp(eta)
        N = np.nan
        err = np.nan    
    elif eta > 20:
        if j == -1/2:
            y = 2 * eta**0.5 / np.sqrt(np.pi)
            N = np.nan
            err = np.nan
        elif j == 1/2:
            y = 4 * eta**(3/2) / 3 / np.sqrt(np.pi)
            N = np.nan
            err = np.nan
        elif j == 3/2:
            y = 8 * eta**(5/2) / 15 / np.sqrt(np.pi)
            N = np.nan
            err = np.nan
    else:
        Nvec = np.linspace(1,Nmax,Nmax)
        for idx, N in enumerate(Nvec):
            a = -4.5 # limits for t
            b = 5.0
            t = np.linspace( a, b, N + 1 ) # generate intervals
            x = np.exp( t - np.exp( -t ) )
            f = x * ( 1 + np.exp( -t ) ) * x ** j / ( 1+ np.exp( x - eta ) )
            y = np.trapz( t, f )
            if N > 1: # test for convergence 
                err = abs( y - y_old )
                if err < tol:
                    break
            y_old = y
            if N == Nmax:
                pass                
                #print( 'Increase the maximum number of iterations.')
        y = -y / spec.gamma( j + 1 )
    if fType == 'script':
        pass
    elif fType == 'roman':
        y = y * np.sqrt(np.pi)/2
    else:
        print("error: unrecognized fType (input 'script' or 'roman'")
    return (y, N, err)



def FD_int_approx( eta, j ):
    import numpy as np
    if j == 1/2:
        if eta <= -2.0:
            y = np.sqrt(np.pi) / 2 * np.exp(eta) * ( 1 - np.exp(eta)/2**1.5 \
              + np.exp(2*eta)/3**1.5 - np.exp(3*eta)/4**1.5 + np.exp(4*eta)/5**1.5 \
              - np.exp(5*eta)/6**1.5 + np.exp(6*eta)/7**1.5 )
        elif eta <= 0:
            y = 0.678091 + 0.53619667*eta + 0.16909748*eta**2  + 0.018780823*eta**3 - 0.0023575446*eta**4 - 0.000639610797*eta**5
        elif eta <= 3.0:
            y = 0.678091 + 0.53638000*eta + 0.16682350*eta**2  + 0.020606700*eta**3 - 0.0060149100*eta**4 + 0.000490398*eta**5
        elif eta <= 10.0:
            y = 0.75706470 + 0.3922888*eta + 0.2705525*eta**2  - 0.016829300*eta**3 + 0.0008258364*eta**4 - 0.00001819771*eta**5
        elif eta <= 1e5:
            y = 2/3 * eta**(3/2) * ( 1 + 1.2337005/eta**2 + 1.0654119/eta**4 + 9.7015185/eta**6 + 242.71502/eta**8 + 12313.691/eta**10 )
        else:
            y = 2/3 * eta**(3/2)
        return (y,)
    else:
        print('only j = +1/2, roman')
        return (np.nan,)




def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



def wavelength_to_rgb(wavelength, gamma=0.8, floor_wave=380, ceiling_wave=750):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength < floor_wave:
        wavelength = floor_wave
    if wavelength > ceiling_wave:
        wavelength = ceiling_wave
    if wavelength < 380:
        R = 0.3
        G = 0.0
        B = 0.3
    elif wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.3
        G = 0.0
        B = 0.0
#    R *= 255
#    G *= 255
#    B *= 255
    return (R, G, B)





def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p
    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """
    import math
    import numpy as np
    if np.isnan(x):
        return "NaN"
    x = float(x)
    if x == 0.:
        return "0." + "0"*(p-1)
    out = []
    if x < 0:
        out.append("-")
        x = -x
    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)
    if n < math.pow(10, p - 1):
        e = e - 1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)
    if abs((n + 1.) * tens - x) <= abs(n * tens - x):
        n = n + 1
    if n >= math.pow(10, p):
        n = n / 10.
        e = e + 1
    m = "%.*g" % (p, n)
    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p - 1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)
    return "".join(out)


def show_plot(figure_id=None):
    import matplotlib.pyplot as plt
    if figure_id is not None:
        fig = plt.figure(num=figure_id)
    else:
        fig = plt.gcf()
    plt.show()
#    plt.pause(1e-9)
#    fig.canvas.manager.window.raise_()

def log_interp1d(xx, yy, kind='linear'):
    import numpy as np
    import scipy.interpolate as scinterp
    logx = np.log10(xx)
    logy = np.log10(yy)
    order = 1
    s = scinterp.InterpolatedUnivariateSpline(logx, logy, k=order)
#    linterp = scinterp.interp1d(logx, logy, kind=kind)
    logterp = lambda zz: np.power(10.0, s(np.log10(zz)))
    return logterp

def raise_window(figname=None):
    import matplotlib.pyplot as plt
    if figname:
        plt.figure(figname)
        cfm = plt.get_current_fig_manager()
        cfm.window.activateWindow()
        cfm.window.raise_()


def prism10widetextarray(nparray,numleftpadspaces,numrightpadspaces,numleftindentspaces,formatstr):
    p10str = ''
    format_base_str = '{:' + formatstr + '}'
    for idx in range(numleftindentspaces):
        p10str += " "
    for val_idx, val in enumerate(nparray):
        if val_idx > 0 and val_idx % 10 == 0:
            p10str += "\n"
            for idx in range(numleftindentspaces):
                p10str += " "
        if val < 0:
            for idx in range(numleftpadspaces-1):
                p10str += " "
        else:
            for idx in range(numleftpadspaces):
                p10str += " "
        p10str += format_base_str.format(val)
        if not val_idx % 10 == 9:
            for idx in range(numrightpadspaces):
                p10str += " "
    return p10str

def nparray2spacedstring(nparray,leftpad,formatstr,rightpad,max_line_width):
    import re
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    spaced_str = np.array2string(
        nparray,
        precision = 5,
        max_line_width = max_line_width,
        separator=',',
        formatter={'float': lambda x: leftpad + format(x, formatstr) + rightpad}
    )
    if(len(leftpad)<2):
        spaced_str = re.sub(r"\[", "", spaced_str)
    else:
        spaced_str = re.sub(r"\[", " ", spaced_str)    
    spaced_str = re.sub(r",|\]", "", spaced_str)
    return spaced_str


def find_nearest(array,value):
    import numpy as np
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]
    
    
def wavelength_nm_to_energy_eV(wavelength_nm):
    h = 6.62606957E-34 # m2kg/s
    e = 1.60217657E-19 # J/eV
    c = 2.99792458E+08 # m/s
    energy_eV = 1e9*h*c/e/wavelength_nm
    return energy_eV
    

def energy_eV_to_wavelength_nm(energy_eV):
    h = 6.62606957E-34 # m2kg/s
    e = 1.60217657E-19 # J/eV
    c = 2.99792458E+08 # m/s
    wavelength_nm = 1e9*h*c/e/energy_eV 
    return wavelength_nm
    
    
def movingAverageFast(x, N):
    import numpy as np
    return np.convolve(x, np.ones((N,))/N, 'same')
    #[(N-1):]
    
def delete_files_with_extension(folder, extensionWithDot, deleteInSubDirectories=False):
    import os
    if deleteInSubDirectories:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(extensionWithDot):
                     print('removing ' + os.path.join(root, file))
                     os.remove(os.path.join(root, file))
    else:
        for file in os.listdir(folder):
            if file.endswith(extensionWithDot):
                print('removing ' + file)
                os.remove(os.path.join(folder, file))
                





###
# Matplotlib 3D axis set_axes_equal
# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
# Mateen Ulhaq
###
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)