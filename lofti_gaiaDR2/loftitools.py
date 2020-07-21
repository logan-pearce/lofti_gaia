import numpy as np
import astropy.units as u
from numpy import tan, arctan, sqrt, cos, sin, arccos

def MonteCarloIt(thing, N = 10000):
    ''' 
    Generate a random sample of size = N from a
    Gaussian centered at thing[0] with std thing[1]
    
    Args:
        thing (tuple, flt): tuple of (value,uncertainty).  Can be either astropy units object 
            or float
        N (int): number of samples
    Returns:
        array: N random samples from a Gaussian.

    Written by Logan Pearce, 2020
    '''
    try:
        out = np.random.normal(thing[0].value,thing[1].value,N)
    except:
        out = np.random.normal(thing[0],thing[1],N)

    return out

def distance(parallax,parallax_error):
    '''
    Computes distance from Gaia parallaxes using the Bayesian method of Bailer-Jones 2015.
    Args:
        parallax, parallax error (flt): parallax in mas
    Returns:
        distance (flt): distance to systems in pc
        sigma (flt): 1-sigma uncertainty in distance

    Written by Logan Pearce, 2018
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
    '''
    Convert from mas/yr -> km/s
     
    Args:
        mas_yr (array): velocity in mas/yr
        plx (tuple,float): parallax, tuple of (plx,plx error)
    Returns:
        km_s (array): velocity in km/s
    
    Written by Logan Pearce, 2019
    '''
    d = distance(*plx)
    # convert mas to km:
    km_s = [((mas_yr[0]*u.mas.to(u.arcsec)*d[0])*u.AU).to(u.km) , ((mas_yr[1]*u.mas.to(u.arcsec)*d[0])*u.AU).to(u.km)]
    # convert yr to s:
    km_s = [(km_s[0].value)*(u.km/u.yr).to(u.km/u.s) , (km_s[1].value)*(u.km/u.yr).to(u.km/u.s)]
    
    return km_s

def mas_to_km(mas,plx):
    '''
    Convert from mas -> km

    Args:
        mas (array, flt): sep in mas
        plx (tuple, flt): parallax, tuple of (plx,plx error) in mas
    Returns:
        array : separation in km

    Written by Logan Pearce, 2019
    '''
    d = distance(*plx)
    # convert mas to km:
    km = [((mas[0]*u.mas.to(u.arcsec)*d[0])*u.AU).to(u.km) , ((mas[1]*u.mas.to(u.arcsec)*d[0])*u.AU).to(u.km)]
    return km

def mas_to_km2(arcsec,dist):
    '''
    Convert from mas -> km using the distance rather than parallax.  Does not
    compute errors on separation.

    Args:
        arcsec (array, flt): sep in arcsec
        dist (array, flt): distance in parsecs
    Returns:
        array : separation in km

    Written by Logan Pearce, 2019
    '''
    # convert to arcsec, and multiply by distance in pc:
    km = (arcsec)*dist # AU
    km = km * 149598073 # km/AU

    return km

def to_polar(RAa,RAb,Deca,Decb):
    ''' Converts RA/Dec [deg] of two binary components into separation and position angle of B relative 
        to A [mas, deg]

    Written by Logan Pearce, 2019
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
    """
    Add elements of an array of things in quadrature
    """
    out = 0
    for t in thing:
        out += t**2
    return np.sqrt(out)

def ComputeChi2(array, model):
    '''
    Cumpute the Chi2 for an array of model predictions and measurements

    Args:
        array (2 x N array): array of observations, row [0] = value, row [1] = errors
        model (1 x N array): array of model predicitons
    
    Returns:
        chi (flt): Chi^2 value

    Written by Logan Pearce, 2020
    '''
    chi = 0
    for i in range(len(array)):
        chi += ( (array[i][0] - model[i]) / array[i][1] ) ** 2
    return chi

######## Fitting tools: ##############
def eccentricity_anomaly(E,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return E - (e*np.sin(E)) - M

def solve(f, M0, e, h, maxnum=50):
    ''' Newton-Raphson solver for eccentricity anomaly
    from https://stackoverflow.com/questions/20659456/python-implementing-a-numerical-equation-solver-newton-raphson
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
    Returns: nextE (float): converged solution for eccentric anomaly
    '''
    import numpy as np
    
    if M0 / (1.-e) - np.sqrt( ( (6.*(1-e)) / e ) ) <= 0:
        E0 = M0 / (1.-e)
    else:
        E0 = (6. * M0 / e) ** (1./3.)
    lastE = E0
    nextE = lastE + 10* h  
    number=0
    while (abs(lastE - nextE) > h) and number < 1001:  
        newY = f(nextE,e,M0) 
        lastE = nextE
        nextE = lastE - newY / (1.-e*np.cos(lastE))
        number=number+1
        if number >= maxnum:
            nextE = mikkola_solve(M0,e)
    return nextE

def danby_solve(f, M0, e, h, maxnum=50):
    ''' Newton-Raphson solver for eccentricity anomaly based on "Danby" method.

    Args: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
        maxnum (int): if it takes more than maxnum iterations,
            use the Mikkola solver instead.
    Returns: 
        nextE (float): converged solution for eccentric anomaly

    Written by Logan Pearce, 2019
    '''
    #f = eccentricity_anomaly
    k = 0.85
    E0 = M0 + np.sign(np.sin(M0))*k*e
    lastE = E0
    nextE = lastE + 10* h 
    number=0
    delta_D = 1
    while (delta_D > h) and number < maxnum+1: 
        fx = f(nextE,e,M0) 
        fp = (1.-e*np.cos(lastE)) 
        fpp = e*np.sin(lastE)
        fppp = e*np.cos(lastE)
        lastE = nextE
        delta_N = -fx / fp
        delta_H = -fx / (fp + 0.5*fpp*delta_N)
        delta_D = -fx / (fp + 0.5*fpp*delta_H + (1./6)*fppp*delta_H**2)
        nextE = lastE + delta_D
        number=number+1
        if number >= maxnum:
            nextE = mikkola_solve(M0,e)
    return nextE


def mikkola_solve(M,e):
    ''' Analytic solver for eccentricity anomaly from Mikkola 1987. Most efficient
        when M near 0/2pi and e >= 0.95.
    Args: 
        M (float): mean anomaly
        e (float): eccentricity
    Returns: 
        eccentric anomaly (flt)

    Written by Logan Pearce, 2019
    '''
    # Constants:
    alpha = (1 - e) / ((4.*e) + 0.5)
    beta = (0.5*M) / ((4.*e) + 0.5)
    ab = np.sqrt(beta**2. + alpha**3.)
    z = np.abs(beta + ab)**(1./3.)

    # Compute s:
    s1 = z - alpha/z
    # Compute correction on s:
    ds = -0.078 * (s1**5) / (1 + e)
    s = s1 + ds

    # Compute E:
    E0 = M + e * ( 3.*s - 4.*(s**3.) )

    # Compute final correction to E:
    sinE = np.sin(E0)
    cosE = np.cos(E0)

    f = E0 - e*sinE - M
    fp = 1. - e*cosE
    fpp = e*sinE
    fppp = e*cosE
    fpppp = -fpp

    dx1 = -f / fp
    dx2 = -f / (fp + 0.5*fpp*dx1)
    dx3 = -f / ( fp + 0.5*fpp*dx2 + (1./6.)*fppp*(dx2**2) )
    dx4 = -f / ( fp + 0.5*fpp*dx3 + (1./6.)*fppp*(dx3**2) + (1./24.)*(fpppp)*(dx3**3) )

    return E0 + dx4

def draw_samples(number, m_tot, d_star, date):
    """
    Draw a set of orbital elements from proability distribution functions.

    Parameters are drawn from the probability distributions:
        a: single value for semi-major axis due to scale and rotate
        const: unif [0,1]
        e: unif [0,1)
        cos(i): unif [0,1]
        w: unif [0,360] deg
        O: single value fixed at 0 due to scale and rotate

    Args: 
        number (int): number of orbits desired to draw elements for. Typically 10000
        m_tot [Msol] (tuple, flt): total system mass[0] and error[1] in solar masses
        d_star [pc] (tuple, flt): distance to system in pc
        date [decimal year] (flt): observation date.  2015.5 for Gaia DR2

    Returns:
        array of samples:
            a [as]: semi-major axis - set at 100 AU inital value
            T [yr]: period (derived from a via Kepler's 3rd law)
            const: constant defining orbital phase of observation
            to [yr]: epoch of periastron passage (derived from orbital phase constant)
            e: eccentricity
            i [rad]: inclination in radians
            w [rad]: arguement of periastron
            O [rad]: longitude of nodes - set at 0 initial value
            m1 [Msol]: total system mass in solar masses
            dist [pc]: distance to system
    
    Written by Logan Pearce, 2018
    """
    m1 = np.random.normal(m_tot[0],m_tot[1],number)
    dist = np.random.normal(d_star[0],d_star[1],number)
    #m1 = m_tot[0]
    #dist = d_star[0]
    # Fixing and initial semi-major axis:
    a_au=100.0
    a_au=np.linspace(a_au,a_au,number)
    T = np.sqrt((np.absolute(a_au)**3)/np.absolute(m1))
    a = a_au/dist #semimajor axis in arcsec

    # Fixing an initial Longitude of ascending node in radians:
    O = np.radians(0.0)  
    O=np.array([O]*number)

    # Randomly generated parameters:
    #to = Time of periastron passage in years:
    const = np.random.uniform(0.0,1.0,number)
    #^ Constant that represents the ratio between (reference epoch minus to) over period.  Because we are scaling
    #semi-major axis, period will also scale, and epoch of periastron passage will change as a result.  This ratio
    #will remain constant however, so we can use it scale both T and to appropriately.
    to = date-(const*T)

    # Eccentricity:
    e = np.random.uniform(0.0,1.0,number)
    # Inclination in radians:
    cosi = np.random.uniform(-1.0,1.0,number)  #Draws sin(i) from a uniform distribution.  Inclination
    # is computed as the arccos of cos(i):
    i = np.arccos(cosi)
    # Argument of periastron in degrees:
    w = np.random.uniform(0.0,360.0,number)
    w = np.radians(w) 
    # collect into array
    samples = np.zeros((10,10000))
    samples[0,:],samples[1,:],samples[2,:],samples[3,:],samples[4,:],samples[5,:], \
        samples[6,:],samples[7,:],samples[8,:],samples[9,:] = a,T,const,to,e,i,w,O,m1,dist
    #samples = np.array([a,T,const,to,e,i,w,O,m1,dist])
    return samples

def scale_and_rotate(X,Y,rho,pa,a,const,m1,dist,d):
    ''' 
    Generates a new semi-major axis, period, epoch of peri passage, and long of peri for each orbit
    given the X,Y plane of the sky coordinates for the orbit at the date of the reference epoch.
    Called by calc_OFTI

    Args:
        X,Y (1 x N array): X and Y model predicted DEC/RA initial position in mas
        rho (tuple, flt): observed separation and error in mas
        pa (tuple, flt): observed position angle and error in deg
        a (1 x N array): initial semi-major axis 
        const (1 x N array): initial orbit fraction
        m1: system mass
        dist: distance to system
        d: observation date (2015.5 for Gaia DR2)
    
    Returns:
        a2 (1 x N array): scaled semi-major axis
        T2 (1 x N array): scaled period
        to2 (1 x N array): scaled periastron passage
        O2 (1 x N array): rotated lon of ascending node

    Written by Logan Pearce, 2018
    '''
    import numpy as np
    
    r_model = np.sqrt((X**2)+(Y**2))
    rho_rand = np.random.normal(rho[0]/1000.,rho[1]/1000.) #This generates a gaussian random to 
    #scale to that takes observational uncertainty into account.  #convert to arcsec
    #rho_rand = rho/1000.

    # scale:
    a2 = a*(rho_rand/r_model) 
    #New period:
    a2_au=a2*dist #convert to AU for period calc:
    T2 = np.sqrt((np.absolute(a2_au)**3)/np.absolute(m1))
    #New epoch of periastron passage
    to2 = d-(const*T2)

    # Rotate:
    PA_model = (np.degrees(np.arctan2(X,-Y))+270)%360 #corrects for difference in zero-point
    #between arctan function and ra/dec projection
    PA_rand = np.random.normal(pa[0],pa[1]) #Generates a random PA within 1 sigma of observation
    #PA_rand = pa
    #New omega value:
    O2=[]
    for PA_i in PA_model:
        if PA_i < 0:
            O2.append((PA_rand-PA_i) + 360.)
        else:
            O2.append(PA_rand-PA_i)
    # ^ This step corrects for the fact that the arctan gives the angle from the +x axis being zero,
    #while for RA/Dec the zero angle is +y axis.  

    #Recompute model with new rotation:
    O2 = np.array(O2)
    O2 = np.radians(O2)
    
    return a2,T2,to2,O2

def calc_XYZ(a,T,to,e,i,w,O,date):
    ''' Compute projected on-sky position only of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point. 
        Called by calc_OFTI

        Args:
            a [as]: semi-major axis
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date

        Returns: 
            X, Y, and Z coordinates [as] where +X is in the reference direction (north) and +Y is east, and +Z
                is towards observer

    Written by Logan Pearce, 2018
    '''
    n = (2*np.pi)/T
    M = n*(date-to)
    nextE = [danby_solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
    E = np.array(nextE)
    # true anomaly:
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    # orbit plane radius in as:
    r = (a*(1.-e**2))/(1.+(e*cos(f)))
    X = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
    Y = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
    Z = r * sin(w+f)*sin(i)
    return X,Y,Z,E

def calc_velocities(a,T,to,e,i,w,O,date,dist,E):
    ''' Compute 3-d velocity of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point.  Uses my eqns derived from Seager 
        Exoplanets Ch2.

        Args:
            a [as]: semi-major axis
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system
            E: eccentricity anomaly computed by calc_XYZ

        Returns: 
            X dot, Y dot, Z dot: three dimensional velocities [km/s]

        Written by Logan Pearce, 2018
    '''
    #from lofti_gaiaDR2.loftifittingtools import to_si, solve
    
    # convert to km:
    a_km = mas_to_km2(a,dist)
    
    # Compute true anomaly:
    n = (2*np.pi)/T
    #M = n*(date-to)
    #nextE = [danby_solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
    #E = np.array(nextE)
    #E = solve(eccentricity_anomaly, M,e, 0.001)
    #r1 = a*(1.-e*cos(E))
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    
    # Compute velocities:
    rdot = ( (n*a_km) / (np.sqrt(1-e**2)) ) * e*sin(f)
    rfdot = ( (n*a_km) / (np.sqrt(1-e**2)) ) * (1 + e*cos(f))
    Xdot = rdot * (cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i)) + \
           rfdot * (-cos(O)*sin(w+f) - sin(O)*cos(w+f)*cos(i))
    Ydot = rdot * (sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i)) + \
           rfdot * (-sin(O)*sin(w+f) + cos(O)*cos(w+f)*cos(i))
    Zdot = ((n*a_km) / (np.sqrt(1-e**2))) * sin(i) * (cos(w+f) + e*cos(w))
    
    Xdot = Xdot*(u.km/u.yr).to((u.km/u.s))
    Ydot = Ydot*(u.km/u.yr).to((u.km/u.s))
    Zdot = Zdot*(u.km/u.yr).to((u.km/u.s))
    return Xdot,Ydot,Zdot

def calc_accel(a,T,to,e,i,w,O,date,dist,E):
    ''' Compute 3-d acceleration of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point. 
        Called by calc_OFTI

        Args:
            a [as]: semi-major axis in as
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
            E: eccentricity anomaly computed by calc_XYZ

        Returns: 
            X ddot, Y ddot, Z ddot: three dimensional accelerations [m/s/yr]

        Written by Logan Pearce, 2018
    '''
    
    #from lofti_gaiaDR2.loftifittingtools import to_si, solve
    # convert to km:
    a_km = mas_to_km2(a,dist)
    # mean motion:
    n = (2*np.pi)/T
    # r and f:
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    r = (a_km*(1-e**2))/(1+e*cos(f))
    # Time derivatives of r, f, and E:
    Edot = n/(1-e*cos(E))
    rdot = e*sin(f)*((n*a_km)/(sqrt(1-e**2)))
    fdot = ((n*(1+e*cos(f)))/(1-e**2))*((sin(f))/sin(E))
    # Second time derivatives:
    Eddot = ((-n*e*sin(f))/(1-e**2))*fdot
    rddot = a_km*e*cos(E)*(Edot**2) + a_km*e*sin(E)*Eddot
    fddot = Eddot*(sin(f)/sin(E)) - (Edot**2)*(e*sin(f)/(1-e*cos(E)))
    # Positional accelerations:
    Xddot = (rddot - r*fdot**2)*(cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i)) + \
            (-2*rdot*fdot - r*fddot)*(cos(O)*sin(w+f) + sin(O)*cos(w+f)*cos(i))
    Yddot = (rddot - r*fdot**2)*(sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i)) + \
            (2*rdot*fdot + r*fddot)*(sin(O)*sin(w+f) + cos(O)*cos(w+f)*cos(i))
    Zddot = sin(i)*((rddot - r*(fdot**2))*sin(w+f) + ((2*rdot*fdot + r*fddot)*cos(w+f)))
    return Xddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr)), Yddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr)), \
                    Zddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr))

def calc_OFTI(parameters,date,rho,pa):
    '''Perform OFTI steps to determine position/velocity/acceleration predictions given
       orbital elements.

        Args:
            Parameters (7 x N array):
                a [as]: semi-major axis
                T [yrs]: period
                to [yrs]: epoch of periastron passage (in same time structure as dates)
                e: eccentricity
                i [rad]: inclination
                w [rad]: argument of periastron
                O [rad]: longitude of nodes
                m1 [Msun]: system mass
                dist [pc]: distance to system in pc
            date [yrs]: observation date
            rho [mas] (tuple, flt): separation and error
            pa [deg] (tuple, flt): position angle and error
        Returns: 
            X, Y, Z positions in plane of the sky [mas],
            X dot, Y dot, Z dot three dimensional velocities [km/s]
            X ddot, Y ddot, Z ddot 3d accelerations in [m/s/yr]

        Written by Logan Pearce, 2018
    '''
    import numpy as np
    import astropy.units as u
    #a,T,const,to,e,i,w,O,m1,dist = parameters
    p = parameters
    # pull values out of array:
    a,T,const,to,e,i,w,O,m1,dist = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]
    # Calculate predicted positions at observation date:
    X1,Y1,Z1,E1 = calc_XYZ(a,T,to,e,i,w,O,date)
    # scale and rotate:
    a2,T2,to2,O2 = scale_and_rotate(X1,Y1,rho,pa,a,const,m1,dist,date)
    # recompute predicted position:
    X2,Y2,Z2,E2 = calc_XYZ(a2,T2,to2,e,i,w,O2,date)
    # convert units:
    X2,Y2,Z2 = (X2*u.arcsec).to(u.mas).value, (Y2*u.arcsec).to(u.mas).value, (Z2*u.arcsec).to(u.mas).value
    # Compute velocities at observation date:
    Xdot,Ydot,Zdot = calc_velocities(a2,T2,to2,e,i,w,O2,date,dist,E2)
    # Compute accelerations at observation date:
    Xddot,Yddot,Zddot = calc_accel(a2,T2,to2,e,i,w,O2,date,dist,E2)
    # Convert to degrees for output:
    i,w,O2 = np.degrees(i),np.degrees(w),np.degrees(O2)
    # put back into output array:
    p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7] = a2,T2,const,to2,e,i,w,O2
    #return X2,Y2,Z2,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2
    return X2,Y2,Z2,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,p

def AcceptOrReject(chi2,chi_min):
    ''' 
    Perform rejection sampling accept/reject decision
    
    Args:
        chi2 (1xN array): array of chi2 values
        chi_min (flt): min chi2 value

    Returns:
        accepted: Array of indicies of accepted
        lnprob: log probability of accepted orbits
        lnrand: log of the random "dice roll" for acceptance

    Written by Logan Pearce, 2020
    '''
    nvals = len(chi2)
    lnprob = -(chi2-chi_min)/2.0
    rand = np.random.uniform(0.0,1.0,nvals)
    accepted = np.where(lnprob > np.log(rand))
    return accepted, lnprob, np.log(rand)

def update_progress(n,max_value):
    ''' 
    Create a progress bar
    
    Args:
        n (int): current count
        max_value (int): ultimate values
    
    '''
    import sys
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    progress = np.round(np.float(n/max_value),decimals=2)
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1.:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}% ({1} of {2}): |{3}|  {4}".format(np.round(progress*100,decimals=1), 
                                                  n, 
                                                  max_value, 
                                                  "#"*block + "-"*(barLength-block), 
                                                  status)
    sys.stdout.write(text)
    sys.stdout.flush()

###  Stats functions  ###
def freedman_diaconis(array):
    '''Compute the optimal number of bins for a 1-d histogram using the Freedman-Diaconis rule of thumb
       Bin width = 2IQR/cuberoot(N)
       Inputs:
           array (arr): flattened array of data
        Returns:
           bin_width (flt): width of bin in optimal binning
           n (int): number of bins
    '''
    import numpy as np
    # Get number of observations:
    N = np.shape(array)[0]
    # Get interquartile range:
    iqr = np.diff(np.quantile(array, q=[.25, .75]))
    bin_width = (2.*iqr)/(N**(1./3.))
    n = int(((np.max(array) - np.min(array)) / bin_width)+1)
    return bin_width, n

def Mode(array):
    '''
    Compute mode of an array
    '''
    # create histogram:
    n, bins = np.histogram(array, freedman_diaconis(array)[1])
    # max bin
    max_bin = np.max(n)
    # find inner/outer bin edges
    bin_inner_edge = np.where(n==max_bin)[0]
    bin_outer_edge = np.where(n==max_bin)[0]+1
    # value in the middle of the highest bin:
    mode=(bins[bin_outer_edge] - bins[bin_inner_edge])/2 + bins[bin_inner_edge]
    return mode[0]

def calc_min_credible_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    From: https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/master/hpd.py
    """
    import numpy as np
    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max

def compute_statistics(array):
    '''
    Compute mode of an array
    '''
    mean = np.mean(array)
    median = np.median(array)
    mode = Mode(array)
    std = np.std(array)
    # 68% CI:
    sorts = np.sort(array)
    frac=0.683
    ci68 = calc_min_credible_interval(sorts,(1-frac))
    # 95% CI:
    frac=0.954
    ci95 = calc_min_credible_interval(sorts,(1-frac))

    return mean, median, mode, std, ci68, ci95

def limit_to_180deg(array):
    '''
    Limit elements of array to [0,180] degrees
    '''
    return array % 180

def orbits_for_plotting(a1,T1,to1,e1,i1,w1,O1,t):
    '''
    Compute X,Y from orbital elements
    '''
    n = (2.*np.pi)/T1
    M = n*(t-to1)
    E = np.array([])
    for M1 in M:
        nextE = danby_solve(eccentricity_anomaly, M1, e1, 0.001)  
        E = np.append(E, nextE)
    A = a1*((cos(O1)*cos(w1))-(sin(O1)*sin(w1)*cos(i1)))
    B = a1*((sin(O1)*cos(w1))+(cos(O1)*sin(w1)*cos(i1)))
    F = a1*((-cos(O1)*sin(w1))-(sin(O1)*cos(w1)*cos(i1)))
    G = a1*((-sin(O1)*sin(w1))+(cos(O1)*cos(w1)*cos(i1)))
    xe = cos(E)-e1
    ye = sqrt(1-e1**2)*sin(E)
    X1 = A*xe + F*ye
    Y1 = B*xe + G*ye
    return X1, Y1

def make_parameter_string(thing,name):
    '''
    Make a string of stats for writing to txt file.
    '''
    string = '   '.join([str(p) for p in [name,thing.mean,thing.median,\
                thing.mode,thing.std,thing.ci68,thing.ci95]])
    return string

