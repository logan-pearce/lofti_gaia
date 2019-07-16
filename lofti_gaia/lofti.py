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

### OFTI fitting tools: ###

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
    from lofti_gaia.lofti import solve
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
    from lofti_gaia.lofti import to_si, solve
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
            a [as]: semi-major axis in mas
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
        Returns: X ddot, Y ddot, Z ddot three dimensional velocities [m/s/yr]
    '''
    import numpy as np
    import astropy.units as u
    from lofti_gaia.lofti import to_si, solve
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    
    # convert to km:
    a_km = to_si(a*1000.,0.,dist)[0]
    # Compute true anomaly:
    n = (2*np.pi)/T
    M = n*(date-to)
    nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
    E = np.array(nextE)
    #E = solve(eccentricity_anomaly, M,e, 0.001)
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
    Zddot = sin(i)*((rddot - r*fdot**2)*sin(w+f) + (2*rdot*fdot+ r*fddot*cos(w+f)))
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

### Plotting tools: ###





### OFTI: ###

def prepareconstraints(source_id1, source_id2):
    """ Convert Gaia astrometric solution into constraints to feed to orbit fitter
        Inputs:
          source_id1 (int): Gaia DR2 source id for primary star
          source_id2 (int): Gaia DR2 source id for secondary star
        Returns:
          deltaRA [mas] (tuple, flt): relative separation[0] and error[1] in 
              RA direction in mas
          deltaDec [mas] (tuple, flt): rel sep[0] and error[1] in Dec direction
          pmRA_kms [km/s] (tuple, flt): relative proper motion[0] and error[1] in RA direction
              in km/s
          pmDec_kms [km/s] (tuple, flt): rel pm[0] and error[1] in Dec direction in kms
          deltarv [km/s] (tuple, flt): relative radial velocity[0] and error[1]. Set to 0 if one 
              or both objects do not have a radial velocity measurement (most common).  Positive rv 
              is defined as *towards the observer* (contrary to rv convention)
          total_pos_velocity [mas/yr] (tuple, flt): total plane-of-sky velocity vector[0] and 
              error[1] in mas/yr
          total_velocity_kms [km/s] (tumple, flt): total velocity[0] and error[1] in km/s, including
              rv if present.
          rho [mas] (tuple, flt): separation[0] and error[1] in mas
          pa [deg] (tuple, flt): position angle[0] and error[1] in degrees east of north
          delta_mag (flt): contrast in magnitudes from primary to secondary star
          d_star [pc] (flt): distance to system in parsecs
    """

    from astroquery.gaia import Gaia
    import astropy.units as u
    from lofti_gaia.lofti import distance, to_polar
    import numpy as np
    # Astroquery throws some warnings we can ignore:
    import warnings
    warnings.filterwarnings("ignore")

    deg_to_mas = 3600000.
    mas_to_deg = 1./3600000.
    
    # Retrieve astrometric solution from Gaia DR2:
    job = Gaia.launch_job("SELECT * FROM gaiadr2.gaia_source WHERE source_id = "+str(source_id1))
    j = job.get_results()

    job = Gaia.launch_job("SELECT * FROM gaiadr2.gaia_source WHERE source_id = "+str(source_id2))
    k = job.get_results()

    # Parallaxes:
    plxa, plxaerr = j[0]['parallax'], j[0]['parallax_error']
    plxb, plxberr = k[0]['parallax'], k[0]['parallax_error']

    # Compute distance:
    db,da = distance(plxb,plxberr),distance(plxa,plxaerr)
    d_star,d_star_err = np.mean([da[0],db[0]]),np.mean([da[1],db[1]])

    # Positions:
    RAa, RAaerr = j[0]['ra'], j[0]['ra_error']*mas_to_deg
    RAb, RAberr = k[0]['ra'], k[0]['ra_error']*mas_to_deg
    Deca, Decaerr = j[0]['dec'], j[0]['dec_error']*mas_to_deg
    Decb, Decberr = k[0]['dec'], k[0]['dec_error']*mas_to_deg

    # Proper motions:
    pmRAa, pmRAaerr = j[0]['pmra'], j[0]['pmra_error']
    pmRAb, pmRAberr = k[0]['pmra'], k[0]['pmra_error']
    pmDeca, pmDecaerr = j[0]['pmdec'], j[0]['pmdec_error']
    pmDecb, pmDecberr = k[0]['pmdec'], k[0]['pmdec_error']

    # Radial velocity, if both objects have a measurement (rare):
    rv = 'no'
    if type(k[0]['radial_velocity']) == np.float64 and type(j[0]['radial_velocity']) == np.float64:
        rv = 'yes'
        rvaarray = np.random.normal(j[0]['radial_velocity'],j[0]['radial_velocity_error'],10000)
        rvbarray = np.random.normal(k[0]['radial_velocity'],k[0]['radial_velocity_error'],10000)
        #deltarv, deltarverr = np.mean(rvaarray - rvbarray), np.std(rvaarray - rvbarray)
        deltarv, deltarverr = j[0]['radial_velocity'] - k[0]['radial_velocity'], np.std(rvaarray - rvbarray)
    else:
        deltarv, deltarverr = 0., 0.

    # Compute relative position and proper motions (monte carlo for errors):
    raa_array = np.random.normal(RAa, RAaerr, 10000)
    rab_array = np.random.normal(RAb, RAberr, 10000)
    deca_array = np.random.normal(Deca, Decaerr, 10000)
    decb_array = np.random.normal(Decb, Decberr, 10000)
    ra_array = (rab_array*deg_to_mas - raa_array*deg_to_mas) * \
                np.cos(np.radians(np.mean([Deca,Decb])))
    deltaRA,deltaRA_err = np.mean(ra_array), np.std(ra_array)
    dec_array = ((decb_array - deca_array)*u.deg).to(u.mas).value
    deltaDec,deltaDec_err = np.mean(dec_array),np.std(dec_array)
    # Compute vel of B relative to A
    pmRA, pmRAerr = (pmRAb - pmRAa), np.sqrt( pmRAberr**2 + pmRAaerr**2 )
    pmDec, pmDecerr = pmDecb - pmDeca, np.sqrt( pmDecaerr**2 + pmDecberr**2 )

    # Convert postions/pms to km:
    h,g = to_si(np.array([deltaRA,deltaDec]),np.array([pmRA,pmDec]),d_star)
    deltaRA_km, deltaDec_km = h[0],h[1]
    pmRA_kms, pmDec_kms = g[0],g[1]
    delta_err_km, pm_err_kms = to_si(
        np.array([ra_array,dec_array] ),
        np.array([np.random.normal( pmRA, pmRAerr, 10000 ), 
                      np.random.normal(pmDec, pmDecerr, 10000)]),
        d_star
        )
    deltaRA_err_km,deltaDec_err_km = np.std( delta_err_km[0] ), np.std( delta_err_km[1] )
    pmRA_err_kms, pmDec_err_kms = np.std( pm_err_kms[0] ), np.std( pm_err_kms[1] )

    # Compute separation/position angle for scale and rotate step:
    rho_array, pa_array = to_polar(raa_array,rab_array,deca_array,decb_array)
    rho, rhoerr  = np.mean(rho_array).value, np.std(rho_array).value
    pa, paerr = np.mean(pa_array).value,np.std(pa_array).value

    # Total plane-of-sky velcocity vector:
    if rv == 'no':
        total_velocity_kms, total_velocity_error_kms = np.sqrt(pmDec_kms**2+pmRA_kms**2), \
          np.sqrt(pmDec_err_kms**2+pmRA_err_kms**2)
    if rv == 'yes':
        total_velocity_kms, total_velocity_error_kms = np.sqrt(pmDec_kms**2+pmRA_kms**2+deltarv**2), \
          np.sqrt(pmDec_err_kms**2+pmRA_err_kms**2+deltarv**2)
    total_pos_velocity, total_pos_velocity_error = np.sqrt(pmDec**2+pmRA**2), np.sqrt(pmDecerr**2+pmRAerr**2)
        

    # Contrast:
    delta_mag = j[0]['phot_g_mean_mag'] - k[0]['phot_g_mean_mag']

    return [deltaRA, deltaRA_err], [deltaDec, deltaDec_err], [pmRA_kms, pmRA_err_kms], \
           [pmDec_kms, pmDec_err_kms], [deltarv, deltarverr], [total_pos_velocity, total_pos_velocity_error], \
           [total_velocity_kms, total_velocity_error_kms], [rho, rhoerr], [pa, paerr], \
           delta_mag, [d_star,d_star_err]

    

def fitorbit(source_id1, source_id2,
                 mass1 = 0,
                 mass2 = 0,
                 d = 2015.5,
                 verbose = False,
                 output_directory = '.',
                 rank = 0,
                 accept_min = 10000
                 ):
    """ 
    Fit orbital parameters to binary stars using only the RA/DEC positions and proper motions from
    Gaia DR2 by inputting the source ids of the two objects and their masses only. 
    Writes accepted orbital parameters to a file.
    
    Parameters:
    -----------
    source_id1, source_id2 : int 
        Gaia DR2 source identifiers, found in the Gaia archive or Simbad.  Fit will be
        of source_id2 relative to source_id1.
    mass1, mass2 : tuple, flt [Msol]
        masses of primary and secondary objects, entered as a tuple with the error.  For example:
        mass1 = (1.0,0.2) is a 1 solar mass star with error of \pm 0.2 solar masses.  If mass1 or mass2 = 0,
        script will prompt user to input a tuple mass.  Default = 0.
    d : flt [decimalyear]
        observation date.  Default = 2015.5, the DR2 obs date.
    verbose : bool
        if set to True, script will print constraints to screen, ask for confrimation before proceeding,
        and print regular updates on number of accepted orbits.  If set to False, script will print a progress bar 
        to screen.  Default = False.
    output_name : str
        directory to write output files to.  If verbose = True, script will prompt for directory, if 
        verbose = False it will write files to current directly unless the name argument is specified.
    rank : int
        if running in parallel processing mode, set this keyword to the rank of each process.  Else it is NA.
    accept_min : int
        when the number of accepted orbits reaches this number, script will terminate

    Returns:
    --------
    output file :
        writes out accepted orbits to a file called name+rank+'_accepted'.  The columns of the file are:
            semi-major axis [arcsec]
            period [yrs]
            epoch of periastron passage [decimalyear]
            eccentricity
            inclination [deg]
            argument of periastron [deg]
            position angle of nodes [deg]
            chi-squared value of the orbit
            probability of orbit generating observations
            random uniform number to determine acceptance

    Notes:
    ------
    Future versions will adapt to new Gaia data releases and additional constraints.  See 
    Pearce et al. 2019 for more information, including a discussion of how to determine if
    the Gaia DR2 solution is of adequate quality to provide meaningful and accurate constraints
    for orbit fitting.
    If you use this package, please cite Pearce et al. 2019.

    Written by Logan A. Pearce, 2019
    """
    
    import numpy as np
    import progressbar
    import time as tm
    
    print('Computing constraints.')
    # Compute constraints:
    deltaRA, deltaDec, pmRA_kms, pmDec_kms, deltarv, total_pos_velocity, total_velocity_kms, \
    rho, pa, delta_mag, d_star = prepareconstraints(source_id1, source_id2)

    if verbose == True:
        print('Finished computing constraints:')
        print('Delta RA, err in mas:', deltaRA[0], deltaRA[1])
        print('Delta Dec, err in mas:', deltaDec[0], deltaDec[1])
        print()
        print('pmRA, err in km/s:',pmRA_kms[0], pmRA_kms[1])
        print('pmDec, err in km/s:',pmDec_kms[0], pmDec_kms[1])
        if deltarv != 0.:
            print('deltaRV, err im km/s (pos towards observer):',deltarv[0], deltarv[1])
        print()
        print('Total relative velocity [km/s]:',total_velocity_kms[0],'+/-',total_velocity_kms[1])
        print('Total plane-of-sky relative velocity [mas/yr]:',total_pos_velocity[0],'+/-',total_pos_velocity[1])
        print()
        print('sep,err [mas]',rho[0],rho[1], 'pa,err [deg]:',pa[0],pa[1])
        print('sep [AU]',(rho[0]/1000)*d_star[0])
        print('sep, err [km]',to_si(rho[0],0,d_star[0]),to_si(rho[1],0,d_star[0]))
        print('D_star',d_star[0],'+\-',d_star[1])
        print('Delta Gmag',delta_mag)
        print()
        yn = input('Proceed? Hit enter to start the fit, n to exit')
        if yn == 'n':
            return None
        else:
            print("Yeehaw let's go")
    
    #################### Begin the fit: ####################
    
    # Get the masses of the objects:
    if mass1 == 0:
        mass = tuple(map(float,input('Enter mass of object 1 and error separated by a space (ex: 1.02 0.2):').split(' ')))
        mA, mAerr = np.float(mass[0]),np.float(mass[1])
    else:
         mA, mAerr = np.float(mass1[0]),np.float(mass1[1])
    if mass2 == 0:
        mass = tuple(map(float,input('Enter mass of object 2 and error separated by a space (ex: 1.02 0.2):').split(' ')))
        mB, mBerr = np.float(mass[0]),np.float(mass[1])
    else:
        mB, mBerr = np.float(mass2[0]),np.float(mass2[1])
        
    m_tot, m_tot_err = mA + mB, np.sqrt((mAerr**2) + (mBerr**2))

    ########### Perform initial run to get initial chi-squared:
    # Draw random orbits:
    a,T,const,to,e,i,w,O,m1,dist = draw_priors(10000, [m_tot, m_tot_err], d_star, d)
    # Compute positions and velocities:
    X,Y,Z,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2 = calc_OFTI(a,T,const,to,e,i,w,O,d,m_tot,dist,rho,pa)
    # Compute chi squared:
    dr = (deltaRA[0] - Y)/(deltaRA[1])
    dd = (deltaDec[0] - X)/(deltaDec[1])
    dxdot = (pmDec_kms[0] - Xdot)/(pmDec_kms[1])
    dydot = (pmRA_kms[0] - Ydot)/(pmRA_kms[1])
    if deltarv != 0.:
        dzdot = (deltarv[0] - Zdot)/(deltarv[1])
        chi = dr**2 + dd**2 + dxdot**2 + dydot**2 + dzdot**2
    else:
        chi = dr**2 + dd**2 + dxdot**2 + dydot**2
    chi_min = np.nanmin(chi)
    if verbose == True:
        print('Chi-min:',chi_min)

    print('Ok, starting loop')
    if verbose == True:
        print('I will write files out to this directory:',output_directory)
        inp = input('Is that right? Hit enter to proceed, n for no: ')
        if inp == 'n':
            output_directory = str(input('Enter path to desired directory: '))
        print('I will be looking for ',accept_min,' orbits.')
        inp = input('Ok? Hit enter to proceed, n for no: ')
        if inp == 'n':
            accept_min = float(input('Enter desired number of orbits: '))
    
    output_file = output_directory + '/accepted_'+str(rank)
    k = open(output_file, 'w')

    # initialize:
    num = 0
    loop_count = 0
    bar = progressbar.ProgressBar(max_value=accept_min)
    start=tm.time()
    
    while num <= accept_min:
        # Draw random orbits:
        a,T,const,to,e,i,w,O,m1,dist = draw_priors(10000, [m_tot, m_tot_err], d_star, d)
        # Compute positions and velocities:
        X,Y,Z,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2 = calc_OFTI(a,T,const,to,e,i,w,O,d,m_tot,dist,rho,pa)
        # Compute chi squared:
        dr = (deltaRA[0] - Y)/(deltaRA[1])
        dd = (deltaDec[0] - X)/(deltaDec[1])
        dxdot = (pmDec_kms[0] - Xdot)/(pmDec_kms[1])
        dydot = (pmRA_kms[0] - Ydot)/(pmRA_kms[1])
        chi = dr**2 + dd**2 + dxdot**2 + dydot**2

        # Accept/reject:
        delta_chi = -(chi-chi_min)/2.0
        A = np.exp(delta_chi)
        rand = np.random.uniform(0.0,1.0,10000)  #The random "dice roll" to determine acceptable probability
        accepted = np.where(A > rand)
    
        # Write to file:
        parameters = np.zeros((10,10000))
        parameters[0,:],parameters[1,:],parameters[2,:],parameters[3,:],parameters[4,:],parameters[5,:], \
              parameters[6,:],parameters[7,:],parameters[8,:],parameters[9,:] = a2,T2,to2,e,i,w,O2,chi,A,rand
        parameters=np.transpose(parameters)
        k = open(output_file, 'a')
        for params in parameters[accepted]:
            string = '   '.join([str(p) for p in params])
            k.write(string + "\n")
        k.close()

        ############### Update chi-min ###################
        new_min =  min(chi)
        #determines the minimum chi from this loop
        if new_min < chi_min and verbose == True:
            chi_min = new_min
            print('Found new chi min: ',chi_min)
            found_new_chi_min = 'yes'
        else:
            found_new_chi_min = 'no'
        #if this minimum chi is less than the previously assigned chi_min, update the chi_min variable
        #to the new value, and write it out to this file. 
	
        if found_new_chi_min == 'yes' and num!=0: 
            ############## Recalculate old accepted orbits with new chi-min for acceptance #######
            dat = np.loadtxt(open(output_file,"rb"),delimiter='   ',ndmin=2)
            a,T,to,e,i,w,O,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]
            q = open(output_file, 'w')
            acc = 0
            for a1,T1,to1,e1,i1,w1,O1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,c,A,dice):
                delta_chi1 = -(c1-chi_min)/2.0
                AA = np.exp(delta_chi1)
                if AA > dice1:
                    string = '   '.join([str(p) for p in [a1,T1,to1,e1,i1,w1,O1,c1,AA,dice1]])
                    q.write(string + "\n")
                    acc += 1
                else:
                    pass
            q.close()
            dat2 = np.loadtxt(open(output_file,"rb"),delimiter='   ',ndmin=2)
            num=dat2.shape[0]
        else:
            pass

        if loop_count%10 == 0:
            dat2 = np.loadtxt(open(output_file,"rb"),delimiter='   ',ndmin=2)
            num=dat2.shape[0]
            if verbose == True:
                print('Loop count rank ',rank,': ',loop_count)
                print("Rank ",rank," has found ",num,"accepted orbits")
                #print('Chi-min:',chi_min)
            else:
                try:
                    bar.update(num)
                except ValueError:
                     pass
	
        loop_count = loop_count + 1  #Iterate the counter
        found_new_chi_min = 'no' #reset the re-evaluator for the next loop

    ##### Finishing up:
    print()
    print('Found ',num,' orbits, finishing up...')
    stop=tm.time()
    time=stop-start
    print('This operation took',time,'seconds')
    print('and',time/3600.,'hours')


def loftiplots():
        
         
