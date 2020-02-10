

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
          ruwe (tuple, flt): renormilized unit weight error.  RUWE <~ 1.2 indicates reliable astrometric
              solution.  
              See https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_ruwe.html
    """

    from astroquery.gaia import Gaia
    import astropy.units as u
    from lofti_gaia.loftifittingtools import distance, to_polar, to_si
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
    
    # Retrieve RUWE for both sources
    job = Gaia.launch_job("SELECT * FROM gaiadr2.ruwe WHERE source_id = "+str(source_id1))
    jruwe = job.get_results()

    job = Gaia.launch_job("SELECT * FROM gaiadr2.ruwe WHERE source_id = "+str(source_id2))
    kruwe = job.get_results()

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
           delta_mag, [d_star,d_star_err], [jruwe['ruwe'],kruwe['ruwe']]

    

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
    output_directory : str
        directory to write output files to.  If verbose = True, script will prompt for directory, if 
        verbose = False it will write files to current directly unless the name argument is specified.
    rank : int
        if running in parallel processing mode, set this keyword to the rank of each process.  Else it is NA.
    accept_min : int
        when the number of accepted orbits reaches this number, script will terminate

    Returns:
    --------
    output files :
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
        writes out a human-readable text file of the constraints it computed from Gaia data, called "constraints.txt".
            deltaRA [mas]: relative RA separation
            deltaDec [mas]: relative DEC separation
            pmRA_kms [km/s]: relative proper motion in RA
            pmDec_kms [km/s]: relative proper motion in DEC
            deltarv [km/s]: relative radial velocity (if applicable)
            total_pos_velocity [mas/yr]: total velocity vector in the plane of the sky
            total_velocity_kms [km.s]: total velocity vector in the plane of the sky 
            rho [mas]: separation
            pa [deg]: position angle
            delta_mag [mag]: contrast in magnitudes
            d_star [pc]: distance
        writes out the above parameters to a machine readable file called "constraints.pkl"

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
    import time as tm
    from lofti_gaia.loftifittingtools import draw_priors, calc_OFTI, to_si, update_progress
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    print('Computing constraints.')
    # Compute constraints:
    deltaRA, deltaDec, pmRA_kms, pmDec_kms, deltarv, total_pos_velocity, total_velocity_kms, \
    rho, pa, delta_mag, d_star, ruwe = prepareconstraints(source_id1, source_id2)

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
        print('RUWE source 1:', ruwe[0][0])
        print('RUWE source 2:', ruwe[1][0])
        print()
        yn = input('Does this look good? Hit enter to start the fit, n to exit: ')
        if yn == 'n':
            return None
        else:
            print("Yeehaw let's go")

    if ruwe[0]>1.2 or ruwe[1]>1.2:
        yn = input('''WARNING: RUWE for one or more of your solutions is greater than 1.2. This indicates 
            that the source might be an unresolved binary or experiencing acceleration 
            during the observation.  Orbit fit results may not be trustworthy.  Do you 
            wish to continue?
            Hit enter to proceed, n to exit: ''')
        if yn == 'n':
            return None
        else:
            pass
        
    
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
    if np.min(deltarv) != 0.:
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

    # Write out constraints to a file:
    
    # Human readable:
    outfile = open(output_directory+'/constraints.txt','w')
    string = 'DeltaRA: '+ str(deltaRA) + '\n'
    string += 'DeltaDEC: '+ str(deltaDec) + '\n'
    string += 'pmRA_kms: '+ str(pmRA_kms) + '\n'
    string += 'pmDEC_kms: '+ str(pmDec_kms) + '\n'
    string += 'deltaRV: '+ str(deltarv) + '\n'
    string += 'Total_Plane_Of_Sky_Vel: '+ str(total_pos_velocity) + '\n'
    string += 'Total_Velocity_kms: '+ str(total_velocity_kms) + '\n'
    string += 'Separation_mas: '+ str(rho) + '\n'
    string += 'PA_deg: '+ str(pa) + '\n'
    string += 'delta_mag: '+ str(delta_mag) + '\n'
    string += 'Distance_pc: '+ str(d_star) + '\n'
    outfile.write(string + "\n")
    outfile.close()
    
    # Machine readable:
    pickle.dump([deltaRA, deltaDec, pmRA_kms, pmDec_kms, deltarv, total_pos_velocity, total_velocity_kms, \
        rho, pa, delta_mag, d_star], open(output_directory+'/constraints.pkl','wb'))

    # Make file to store output:
    output_file = output_directory + '/accepted_'+str(rank)
    k = open(output_file, 'w')
    k.write('# semimajoraxis[arcsec]    period[yrs]    t_o[yr]    ecc    incl[deg]    argofperiastron[deg]    posangleofnodes[deg]\
        chisquaredvalue    proboforbit    randnum' + "\n")
    k.close()

    # initialize:
    num = 0
    loop_count = 0
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
        if np.min(deltarv) != 0.:
            dzdot = (deltarv[0] - Zdot)/(deltarv[1])
            chi = dr**2 + dd**2 + dxdot**2 + dydot**2 + dzdot**2
        else:
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
        # Write out to text file:
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
                update_progress(num,accept_min)
	
        loop_count = loop_count + 1  #Iterate the counter
        found_new_chi_min = 'no' #reset the re-evaluator for the next loop

    ##### Finishing up:
    print()
    print('Found ',num,' orbits, finishing up...')
    stop=tm.time()
    time=stop-start
    print('This operation took',time,'seconds')
    print('and',time/3600.,'hours')


def makeplots(input_directory,
                  rank = 0,
                  Collect_into_one_file = False,
                  limit = 0.,
                  roll_w = False,
                  plot_posteriors = True,
                  plot_orbit_plane = True,
                  plot_3d = True,
                  axlim = 6
              ):
    """ 
    Produce plots and summary statistics for the output from lofti.fitorbit.
    
    Parameters:
    -----------
    input_directory : str
        Gaia DR2 source identifiers, found in the Gaia archive or Simbad.  Fit will be
        of source_id2 relative to source_id1.
    rank : int
        Set this parameter to iterate through processes if running on multiple threads
    Collect_into_one_file : bool
        Set to true if running on multiple process and the script did not terminate on its own.  This will
        tell the script to collect output from each multiple process and put into one file.
    limit : int [au]
        Sometimes semi-major axis posteriors will have very long tails.  If you wish to truncate the sma histogram 
        at some value for clarity, set the limit parameter to that value.
    roll_w : bool
        I you wish to have arg of periastron wrap around zero, set this to True
    plot_posteriors : bool
        set to True to make posterior histogram plots of X, Y, Z, dotX, dotY, dotZ, ddotX, ddotY, ddotZ
    plot_orbit_plane : bool
        set to True to generate plot of 100 random orbits from the posterior in XY plane, XZ plane, and YZ plane
    plot_3d: bool
        set to True to generare a 3D plot of 20 random orbits from the posterior.
    axlim: flt [arcsec]
        if plot_orbits = True or plot_3d = True, set this parameter to set the axis limits (in arcsec) for the plots

    Returns:
    --------
    output files :
        stats: summary statistics for each orbital parameter + periastron distance, including:
            Mean    Std    Mode    68% Min Cred Int    95% Min Cred Int
        hist.pdf: 1d histogram of orbital parameter posteriors
        observable_posteriors (if plot_posteriors = True): directory containing 1d histograms of 
            posteriors for X, Y, Z, dotX, dotY, dotZ, ddotX, ddotY, ddotZ
        orbits.png (if plot_orbits = True): plot of a selection of 100 random orbits from fitorbit posterior in 
            RA/DEC, colored by orbital phase
        orbits_yz.png, orbits_xz.png (if plot_orbits = True): plots of the same 100 orbits in YZ and XZ planes
        orbits_3d.png (if plot_3d = True): 3d plot of 20 random orbits from posterior

    Notes:
    ------
    These are suggested summary stats and plots.  For more versatility you can use
    the fitorbit output with your own plotting scheme.
    If you use this package, please cite Pearce et al. 2019.

    Written by Logan A. Pearce, 2019
    """ 

    import numpy as np
    import pickle
    from lofti_gaia.loftiplots import write_stats, plot_1d_hist, plot_observables_hist, plot_orbits3d, plot_orbits
    import os
    
    files = input_directory + '/accepted_'+str(rank)

    if Collect_into_one_file == True:
        q = open(files, 'w')
        ## Prepare fitter output:
        # Collect into one file:
        for ind in range(size):
            # Collect all the outputs into one file:
            dat = np.loadtxt(open(files+'_'+str(ind),"rb"),delimiter='   ',ndmin=2)
            a,T,to,e,i,w,O,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]
            chi_min = np.min(c)
            q = open(files, 'a')
            for a1,T1,to1,e1,i1,w1,O1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,c,A,dice):
                string = '   '.join([str(p) for p in [a1,T1,to1,e1,i1,w1,O1,c1,A1,dice1]])
                q.write(string + "\n")
            q.close()

        # Reperform accept/reject step with the min chi-squared from all processes:
        dat = np.loadtxt(open(files,"rb"),delimiter='   ',ndmin=2)
        a,T,to,e,i,w,O,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]

        chi_min = np.min(c)
        print('Minimum chi^2 found: ',chi_min)
    
    dat = np.loadtxt(open(files,"rb"),delimiter='   ',ndmin=2)
    num=dat.shape[0]

    # Read in final parameter arrays:
    a,T,to,e,i_deg,w_deg,O_deg,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]
    i,w,O = np.radians(i_deg),np.radians(w_deg),np.radians(O_deg)
    
    # Read in observational constraints:
    infile = open(input_directory+"/constraints.pkl",'rb')
    deltaRA, deltaDec, pmRA_kms, pmDec_kms, deltarv, total_pos_velocity, total_velocity_kms, rho, pa, delta_mag, d_star = pickle.load(infile)
    date = 2015.5
    
    a_au=a*d_star[0]
    periastron = (1.-e)*a_au

    # If desired, truncate the semi-major axis histogram:
    if limit != 0.:
        a_au2 = a_au[np.where(a_au<limit)]
        to2 = to[np.where(a_au<limit)]
        T2 = T[np.where(a_au<limit)]
        periastron2 = periastron[np.where(a_au<limit)]
    else:
        a_au2 = a_au
        to2 = to
        T2 = T
        periastron2 = periastron

    # To center arg of periastron on 180 deg instead of 0:
    if roll_w == True:
        w_temp = w_deg.copy()
        for j in range(len(w_deg)):
            if w_temp[j] > 180:
                w_temp[j] = w_temp[j] - 360.
    else:
        w_temp = w_deg

    O_temp = O_deg.copy()%360
    for j in range(len(O_deg)):
        if O_temp[j] > 180:
            O_temp[j] = O_temp[j] - 360.
        else:
            pass

    plot_params_names = [r"$a \; (AU)$",r"$e$",r"$ i \; (deg)$",r"$ \omega \; (deg)$",r"$\Omega \; (deg)$",r"$T_0 \; (yr)$",\
                         r"$a\,(1-e) \; (AU)$"]
                         
    print('Writing out stats')
    stats_name = input_directory+'/stats'
    write_stats([a_au,e,i_deg,w_temp,O_temp,to,periastron],plot_params_names,stats_name)

    print('Making histograms')
    output_name = input_directory+"/hists.png"
    plot_1d_hist([a_au2,e,i_deg,w_temp,O_deg,to2,periastron],plot_params_names,output_name,50,tick_fs = 25,
                     label_fs = 30,label_x_x=0.5, label_x_y = -0.3)

    if plot_posteriors == True:
        print('Plotting observable posteriors')
        os.system('mkdir '+str(input_directory)+'/observable_posteriors')
        output_name = input_directory + '/observable_posteriors/'
        plot_observables_hist(a,T,to,e,i,w,O,date,d_star[0],output_name)

    if plot_orbit_plane == True:
        print('Plotting orbits')
        # Select random orbits from sample:
        if a.shape[0] >= 100:
            size = 100
        else:
            size = a.shape[0]-1

        index = np.random.choice(range(0,dat.shape[0]),replace=False,size=size)
        a1,T1,to1,e1,i1,w1,O1 = a[index],T[index],to[index],e[index],i[index],w[index],O[index]

        # Plot those orbits:
        # RA/Dec plane:
        print('XY plane')
        output_name = input_directory+"/orbits"
        plot_orbits(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = axlim, ticksize = 15, 
                            labelsize = 20)

        # X/Z plane:
        print('XZ plane')
        output_name = input_directory+"/orbits_xz"
        plot_orbits(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = axlim, plane = 'xz')

        # Y/Z plane:
        print('YZ plane')
        output_name = input_directory+"/orbits_yz"
        plot_orbits(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = axlim, plane = 'yz')


    if plot_3d == True:
        # Select only 20 random orbits from sample:
        if a.shape[0] >= 20:
            size = 20
        else:
            size = a.shape[0]-1

        index = np.random.choice(range(0,dat.shape[0]),replace=False,size=size)
        a1,T1,to1,e1,i1,w1,O1 = a[index],T[index],to[index],e[index],i[index],w[index],O[index]

        print('3D')
        output_name = input_directory+"/orbits_3d"
        plot_orbits3d(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = axlim)
    
        
         
