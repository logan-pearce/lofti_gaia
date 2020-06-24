import numpy as np
import astropy.units as u

def MonteCarloIt(thing, N = 10000):
    ''' Generate a random sample of size = N from a
        Gaussian centered at thing[0] with std thing[1]
        Parameters
        ----------
        thing : tuple
            tuple of (value,uncertainty).  Can be either astropy units object 
            or float
        N : int
            number of samples
        Returns
        -------
        array of N random samples from a Gaussian.
    '''
    try:
        out = np.random.normal(thing[0].value,thing[1].value,N)
    except:
        out = np.random.normal(thing[0],thing[1],N)

    return out

def distance(parallax,parallax_error):
    '''Computes distance from Gaia parallaxes using the Bayesian method of Bailer-Jones 2015.
    Input: parallax [mas], parallax error [mas]
    Output: distance [pc], 1-sigma uncertainty in distance [pc]
    '''

    # Compute most probable distance:
    L=1350 #parsecs
    # Convert to arcsec:
    parallax, parallax_error = parallax/1000., parallax_error/1000.
    # establish the coefficients of the mode-finding polynomial:
    coeff = np.array([(1./L),(-2),((parallax)/((parallax_error)**2)),-(1./((parallax_error)**2))])
    # use numpy to find the roots:
    g = np.roots(coeff)
    # Find the number of real roots:
    reals = np.isreal(g)
    realsum = np.sum(reals)
    # If there is one real root, that root is the  mode:
    if realsum == 1:
        gd = np.real(g[np.where(reals)[0]])
    # If all roots are real:
    elif realsum == 3:
        if parallax >= 0:
            # Take the smallest root:
            gd = np.min(g)
        elif parallax < 0:
            # Take the positive root (there should be only one):
            gd = g[np.where(g>0)[0]]
    
    # Compute error on distance from FWHM of probability distribution:
    from scipy.optimize import brentq
    rmax = 1e6
    rmode = gd[0]
    M = (rmode**2*np.exp(-rmode/L)/parallax_error)*np.exp((-1./(2*(parallax_error)**2))*(parallax-(1./rmode))**2)
    lo = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), 0.001, rmode)
    hi = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), rmode, rmax)
    fwhm = hi-lo
    # Compute 1-sigma from FWHM:
    sigma = fwhm/2.355
            
    return gd[0],sigma

def masyr_to_kms(mas_yr,plx):
    '''Convert from mas/yr -> km/s
        Input: 
         mas_yr (array) [mas/yr]: velocity in mas/yr
         plx (tuple,float) [mas]: parallax, tuple of (plx,plx error)
        Returns:
         km_s (array) [km/s]: velocity in km/s
    '''
    d = distance(*plx)
    # convert mas to km:
    km_s = [((mas_yr[0]*u.mas.to(u.arcsec)*d[0])*u.AU).to(u.km) , ((mas_yr[1]*u.mas.to(u.arcsec)*d[0])*u.AU).to(u.km)]
    # convert yr to s:
    km_s = [(km_s[0].value)*(u.km/u.yr).to(u.km/u.s) , (km_s[1].value)*(u.km/u.yr).to(u.km/u.s)]
    
    return km_s

def mas_to_km(mas,plx):
    '''Convert from mas/yr -> km/s
        Input: 
         mas_yr (array) [mas/yr]: velocity in mas/yr
         plx (tuple,float) [mas]: parallax, tuple of (plx,plx error)
        Returns:
         km_s (array) [km/s]: velocity in km/s
    '''
    d = distance(*plx)
    # convert mas to km:
    km = [((mas[0]*u.mas.to(u.arcsec)*d[0])*u.AU).to(u.km) , ((mas[1]*u.mas.to(u.arcsec)*d[0])*u.AU).to(u.km)]
    
    return km

def to_polar(RAa,RAb,Deca,Decb):
    ''' Converts RA/Dec [deg] of two binary components into separation and position angle of B relative 
        to A [mas, deg]
    '''
    
    dRA = (RAb - RAa) * np.cos(np.radians(np.mean([Deca,Decb])))
    dRA = (dRA*u.deg).to(u.mas)
    dDec = (Decb - Deca)
    dDec = (dDec*u.deg).to(u.mas)
    r = np.sqrt( (dRA ** 2) + (dDec ** 2) )
    p = (np.degrees( np.arctan2(dDec.value,-dRA.value) ) + 270.) % 360.
    p = p*u.deg
    
    return r, p

def add_in_quad(thing):
    out = 0
    for t in thing:
        out += t**2
    return np.sqrt(out)

def test(atribute):
    pass