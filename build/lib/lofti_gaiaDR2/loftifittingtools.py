### Observation tools: ###
def distance(parallax,parallax_error):
    '''Computes distance from Gaia parallaxes using the Bayesian method of Bailer-Jones 2015.
    Input: parallax [mas], parallax error [mas]
    Output: distance [pc], 1-sigma uncertainty in distance [pc]
    '''
    import numpy as np
    
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

def to_si(mas,mas_yr,d):
    '''Convert from mas -> km and mas/yr -> km/s
        Input: 
         mas (array) [mas]: distance in mas
         mas_yr (array) [mas/yr]: velocity in mas/yr
         d (float) [pc]: distance to system in parsecs
        Returns:
         km (array) [km]: distance in km
         km_s (array) [km/s]: velocity in km/s
    '''
    import astropy.units as u
    
    km = ((mas*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = ((mas_yr*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = (km_s.value)*(u.km/u.yr).to(u.km/u.s)
    
    return km.value,km_s

def to_polar(RAa,RAb,Deca,Decb):
    ''' Converts RA/Dec [deg] of two binary components into separation and position angle of B relative 
        to A [mas, deg]
    '''
    import numpy as np
    import astropy.units as u
    
    dRA = (RAb - RAa) * np.cos(np.radians(np.mean([Deca,Decb])))
    dRA = (dRA*u.deg).to(u.mas)
    dDec = (Decb - Deca)
    dDec = (dDec*u.deg).to(u.mas)
    r = np.sqrt( (dRA ** 2) + (dDec ** 2) )
    p = (np.degrees( np.arctan2(dDec.value,-dRA.value) ) + 270.) % 360.
    p = p*u.deg
    
    return r, p

#### Fitting tools ####
def eccentricity_anomaly(E,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return E - (e*np.sin(E)) - M

def solve(f, M0, e, h):
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
        if number >= 1000:
            nextE = float('NaN')
    return nextE

def draw_priors(number, m_tot, d_star, d):
    """Draw a set of orbital elements from proability distribution functions.
        Input: 
            number (int): number of orbits desired to draw elements for. Typically 10000
            m_tot [Msol] (tuple, flt): total system mass[0] and error[1] in solar masses
            d_star [pc] (tuple, flt): distance to system in pc
            d [decimal year] (flt): observation date.  2015.5 for Gaia DR2
        Returns:
            a [as]: semi-major axis - set at 100 AU inital value
            T [yr]: period
            const: constant defining orbital phase of observation
            to [yr]: epoch of periastron passage
            e: eccentricity
            i [rad]: inclination in radians
            w [rad]: arguement of periastron
            O [rad]: longitude of nodes - set at 0 initial value
            m1 [Msol]: total system mass in solar masses
            dist [pc]: distance to system
    """
    import numpy as np
    
    #m1 = np.random.normal(m_tot[0],m_tot[1],number)
    #dist = np.random.normal(d_star[0],d_star[1],number)
    m1 = m_tot[0]
    dist = d_star[0]
    # Fixing and initial semi-major axis:
    a_au=100.0
    a_au=np.linspace(a_au,a_au,number)
    T = np.sqrt((np.absolute(a_au)**3)/np.absolute(m1))
    a = a_au/dist #semimajor axis in arcsec

    # Fixing an initial Longitude of ascending node in radians:
    O = np.radians(0.0)  
    O=[O]*number

    # Randomly generated parameters:
    #to = Time of periastron passage in years:
    const = np.random.uniform(0.0,1.0,number)
    #^ Constant that represents the ratio between (reference epoch minus to) over period.  Because we are scaling
    #semi-major axis, period will also scale, and epoch of periastron passage will change as a result.  This ratio
    #will remain constant however, so we can use it scale both T and to appropriately.
    to = d-(const*T)

    # Eccentricity:
    e = np.random.uniform(0.0,1.0,number)
    # Inclination in radians:
    cosi = np.random.uniform(-1.0,1.0,number)  #Draws sin(i) from a uniform distribution.  Inclination
    # is computed as the arccos of cos(i):
    i = np.arccos(cosi)
    # Argument of periastron in degrees:
    w = np.random.uniform(0.0,360.0,number)
    w = np.radians(w) #convert to radians for calculations
    return a,T,const,to,e,i,w,O,m1,dist

def scale_and_rotate(X,Y,rho,pa,a,const,m1,dist,d):
    ''' Generates a new semi-major axis, period, epoch of peri passage, and long of peri for each orbit
        given the X,Y plane of the sky coordinates for the orbit at the date of the reference epoch
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
        Inputs:
            a [as]: semi-major axis in mas
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
        Returns: X, Y, and Z coordinates [as] where +X is in the reference direction (north) and +Y is east, and +Z
            is towards observer
    '''
    import numpy as np
    from lofti_gaiaDR2.loftifittingtools import solve
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    
    n = (2*np.pi)/T
    M = n*(date-to)
    nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
    E = np.array(nextE)
    #E = solve(eccentricity_anomaly, M,e, 0.001)
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    # orbit plane radius in as:
    r = (a*(1.-e**2))/(1.+(e*cos(f)))
    X = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
    Y = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
    Z = r * sin(w+f)*sin(i)
    return X,Y,Z

def calc_velocities(a,T,to,e,i,w,O,date,dist):
    ''' Compute 3-d velocity of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point.  Uses my eqns derived from Seager 
        Exoplanets Ch2.
        Inputs:
            a [as]: semi-major axis in mas
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            m_tot [Msol]: total system mass
        Returns: X dot, Y dot, Z dot three dimensional velocities [km/s]
    '''
    import numpy as np
    import astropy.units as u
    from lofti_gaiaDR2.loftifittingtools import to_si, solve
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    
    # convert to km:
    a_km = to_si(a*1000.,0.,dist)
    a_km = a_km[0]
    
    # Compute true anomaly:
    n = (2*np.pi)/T
    M = n*(date-to)
    nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
    E = np.array(nextE)
    #E = solve(eccentricity_anomaly, M,e, 0.001)
    r1 = a*(1.-e*cos(E))
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

def calc_accel(a,T,to,e,i,w,O,date,dist):
    ''' Compute 3-d acceleration of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point.  
        Inputs:
            a [as]: semi-major axis in as
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
        Returns: X ddot, Y ddot, Z ddot three dimensional accelerations [m/s/yr]
    '''
    import numpy as np
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    import astropy.units as u
    from lofti_gaiaDR2.loftifittingtools import to_si, solve
    # convert to km:
    a_mas = a*u.arcsec.to(u.mas)
    try:
        a_mas = a_mas.value
    except:
        pass
    a_km = to_si(a_mas,0.,dist)[0]
    # Compute true anomaly:
    n = (2*np.pi)/T
    M = n*(date-to)
    try:
        nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
    except:
        E = solve(eccentricity_anomaly, M,e, 0.001)
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

def calc_OFTI(a,T,const,to,e,i,w,O,d,m1,dist,rho,pa):
    '''Perform OFTI steps to determine position/velocity/acceleration predictions given
       orbital elements.
        Inputs:
            a [as]: semi-major axis in mas
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
            rho [mas] (tuple, flt): separation and error
            pa [deg] (tuple, flt): position angle and error
        Returns: 
            X, Y, Z positions in plane of the sky [mas],
            X dot, Y dot, Z dot three dimensional velocities [km/s]
            X ddot, Y ddot, Z ddot 3d accelerations in [m/s/yr]
    '''
    import numpy as np
    import astropy.units as u
    
    # Calculate predicted positions at observation date:
    X1,Y1,Z1 = calc_XYZ(a,T,to,e,i,w,O,d)
    # scale and rotate:
    a2,T2,to2,O2 = scale_and_rotate(X1,Y1,rho,pa,a,const,m1,dist,d)
    # recompute predicted position:
    X2,Y2,Z2 = calc_XYZ(a2,T2,to2,e,i,w,O2,d)
    # convert units:
    X2,Y2,Z2 = (X2*u.arcsec).to(u.mas).value, (Y2*u.arcsec).to(u.mas).value, (Z2*u.arcsec).to(u.mas).value
    # Compute velocities at observation date:
    Xdot,Ydot,Zdot = calc_velocities(a2,T2,to2,e,i,w,O2,d,dist)
    # Compute accelerations at observation date:
    Xddot,Yddot,Zddot = calc_accel(a2,T2,to2,e,i,w,O2,d,dist)
    # Convert to degrees:
    i,w,O2 = np.degrees(i),np.degrees(w),np.degrees(O2)
    return X2,Y2,Z2,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2

def update_progress(n,max_value):
    import sys
    import time
    import numpy as np
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
