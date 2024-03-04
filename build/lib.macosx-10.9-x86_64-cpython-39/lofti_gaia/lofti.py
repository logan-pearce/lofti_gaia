import astropy.units as u
import numpy as np
from lofti_gaia.loftitools import *
from lofti_gaia.cFunctions import calcOFTI_C
#from loftitools import *
import pickle
import time
import matplotlib.pyplot as plt
# Astroquery throws some warnings we can ignore:
import warnings
warnings.filterwarnings("ignore")

'''This module obtaines measurements from Gaia EDR3 (Gaia DR2 is also available as a secondary option) and runs through the LOFTI Gaia/OFTI 
wide stellar binary orbit fitting technique.
'''

class Fitter(object):
    '''Initialize the Fitter object for the binary system, and compute observational constraints 
    to be used in the orbit fit.  User must provide Gaia source ids, tuples of mass estimates for 
    both objects, specify the number of desired orbits in posterior sample.  Fit will be
    for object 2 relative to object 1.

    Attributes are tuples of (value,uncertainty) unless otherwise indicated.  Attributes
    with astropy units are retrieved from Gaia archive, attributes without units are
    computed from Gaia values.  All relative values are for object 2 relative to object 1.

    Args:
        sourceid1, sourceid2 (int): Gaia source ids for the two objects, fit will be for motion of \
            object 2 relative to object 1
        mass1, mass2 (tuple, flt): tuple os mass estimate for object 1 and 2, of the form (value, uncertainty)
        Norbits (int): Number of desired orbits in posterior sample.  Default = 100000
        results_filename (str): Filename for fit results files.  If none, results will be written to files \
            named FitResults.yr.mo.day.hr.min.s
        astrometry (dict): User-supplied astrometric measurements. Must be dictionary or table or pandas dataframe with\
            column names "sep,seperr,pa,paerr,dates" or "ra,raerr,dec,decerr,dates". May be same as the rv table. \
            Sep, deltaRA, and deltaDEC must be in arcseconds, PA in degrees, dates in decimal years. \
            Default = None
        user_rv (dict): User-supplied radial velocity measurements. Must be dictionary or table or pandas dataframe with\
            column names "rv,rverr,rv_dates". May be same as the astrometry table. Default = None.
        catalog (str): name of Gaia catalog to query. Default = 'gaiaedr3.gaia_source' 
        ruwe1, ruwe2 (flt): RUWE value from Gaia archive
        ref_epoch (flt): reference epoch in decimal years. For Gaia DR2 this is 2015.5, for Gaia EDR3 it is 2016.0
        plx1, plx2 (flt): parallax from Gaia in mas
        RA1, RA2 (flt): right ascension from Gaia; RA in deg, uncertainty in mas
        Dec1, Dec2 (flt): declination from Gaia; Dec in deg, uncertainty in mas
        pmRA1, pmRA2 (flt): proper motion in RA in mas yr^-1 from Gaia
        pmDec1, pmDec2 (flt): proper motion in DEC in mas yr^-1 from Gaia
        rv1, rv2 (flt, optional): radial velocity in km s^-1 from Gaia
        rv (flt, optional): relative RV of 2 relative to 1, if both are present in Gaia
        plx (flt): weighted mean parallax for the binary system in mas
        distance (flt): distance of system in pc, computed from Gaia parallax using method \
            of Bailer-Jones et. al 2018.
        deltaRA, deltaDec (flt): relative separation in RA and Dec directions, in mas
        pmRA, pmDec (flt): relative proper motion in RA/Dec directions in km s^-1
        sep (flt): total separation vector in mas
        pa (flt): postion angle of separation vector in degrees from North
        sep_au (flt): separation in AU
        sep_km (flt): separation in km
        total_vel (flt): total velocity vector in km s^-1.  If RV is available for both, \
            this is the 3d velocity vector; if not it is just the plane of sky velocity. 
        total_planeofsky_vel (flt): total velocity in the plane of sky in km s^-1. \
            In the absence of RV this is equivalent to the total velocity vector.
        deltaGmag (flt): relative contrast in Gaia G magnitude.  Does not include uncertainty.
        inflateProperMOtionError (flt): an optional factor to mulitply default gaia proper motion error by.

    Written by Logan Pearce, 2020
    '''
    def __init__(self, sourceid1, sourceid2, mass1, mass2, Norbits = 100000, \
        results_filename = None, 
        astrometry = None,
        user_rv = None,
        catalog = 'gaiaedr3.gaia_source',
        inflateProperMotionError=1
        ):
        
        self.sourceid1 = sourceid1
        self.sourceid2 = sourceid2
        try:
            self.mass1 = mass1[0]
            self.mass1err = mass1[1]
            self.mass2 = mass2[0]
            self.mass2err = mass2[1]
            self.mtot = [self.mass1 + self.mass2, np.sqrt((self.mass1err**2) + (self.mass2err**2))]
        except:
            raise ValueError('Masses must be tuples of (value,error), ex: mass1 = (1.0,0.05)')
        self.Norbits = Norbits
        if not results_filename:
            self.results_filename = 'FitResults.'+time.strftime("%Y.%m.%d.%H.%M.%S")+'.txt'
            self.stats_filename = 'FitResults.Stats.'+time.strftime("%Y.%m.%d.%H.%M.%S")+'.txt'
        else:
            self.results_filename = results_filename
            self.stats_filename = results_filename+'.Stats.txt'

        self.astrometry = False
        # check if user supplied astrometry:
        if astrometry is not None:
            # if so, set astrometric flag to True:
            self.astrometry = True
            # store observation dates:
            self.astrometric_dates = astrometry['dates']
            # if in sep/pa, convert to ra/dec:
            if 'sep' in astrometry:
                try:
                    astr_ra = [MonteCarloIt([astrometry['sep'][i],astrometry['seperr'][i]]) * \
                        np.sin(np.radians(MonteCarloIt([astrometry['pa'][i],astrometry['paerr'][i]]))) \
                        for i in range(len(astrometry['sep']))]
                    astr_dec = [MonteCarloIt([astrometry['sep'][i],astrometry['seperr'][i]]) * \
                        np.cos(np.radians(MonteCarloIt([astrometry['pa'][i],astrometry['paerr'][i]]))) \
                        for i in range(len(astrometry['sep']))]

                    self.astrometric_ra = np.array([
                        [np.mean(astr_ra[i]) for i in range(len(astrometry['sep']))],
                        [np.std(astr_ra[i]) for i in range(len(astrometry['sep']))]
                        ])
                    self.astrometric_dec = np.array([
                        [np.mean(astr_dec[i]) for i in range(len(astrometry['sep']))],
                        [np.std(astr_dec[i]) for i in range(len(astrometry['sep']))]
                        ])
                except:
                    raise ValueError('Astrometry keys not recognized. Please provide dictionary or table or pandas dataframe with\
                     column names "sep,seperr,pa,paerr,dates" or "ra,raerr,dec,decerr,dates"')
            elif 'ra' in astrometry:
                # else store the ra/dec as attributes:
                try:
                    self.astrometric_ra = np.array([astrometry['ra'], astrometry['raerr']])
                    self.astrometric_dec = np.array([astrometry['dec'], astrometry['decerr']])
                except:
                    raise ValueError('Astrometry keys not recognized. Please provide dictionary or table or pandas dataframe with\
                     column names "sep,seperr,pa,paerr,dates" or "ra,raerr,dec,decerr,dates"')
            else:
                raise ValueError('Astrometry keys not recognized. Please provide dictionary or table or pandas dataframe with\
                     column names "sep,seperr,pa,paerr,dates" or "ra,raerr,dec,decerr,dates"')

        # Check if user supplied rv:
        self.use_user_rv = False
        if user_rv is not None:
            # set user rv flag to true:
            self.use_user_rv = True
            try:
                # set attributes; multiply rv by -1 due to difference in coordinate systems:
                self.user_rv = np.array([user_rv['rv']*-1,user_rv['rverr']])
                self.user_rv_dates = np.array(user_rv['rv_dates'])
            except:
                raise ValueError('RV keys not recognized.  Please use column names "rv,rverr,rv_dates"')
        self.catalog = catalog

        # Get Gaia measurements, compute needed constraints, and add to object:
        self.PrepareConstraints(catalog=self.catalog,inflateFactor=inflateProperMotionError)
    
    def edr3ToICRF(self,pmra,pmdec,ra,dec,G):
        ''' Corrects for biases in proper motion. The function is from https://arxiv.org/pdf/2103.07432.pdf

        Args:
            pmra,pmdec (float): proper motion
            ra, dec (float): right ascension and declination
            G (float): G magnitude

        Written by Sam Christian, 2021
        '''
        if G>=13:
            return pmra , pmdec
        import numpy as np
        def sind(x):
            return np.sin(np.radians(x))
        def cosd(x):
            return np.cos(np.radians(x))
        table1="""
        0.0 9.0 9.0 9.5 9.5 10.0 10.0 10.5 10.5 11.0 11.0 11.5 11.5 11.75 11.75 12.0 12.0 12.25 12.25 12.5 12.5 12.75 12.75 13.0
        18.4 33.8 -11.3 14.0 30.7 -19.4 12.8 31.4 -11.8 13.6 35.7 -10.5 16.2 50.0 2.1 19.4 59.9 0.2 21.8 64.2 1.0 17.7 65.6 -1.9 21.3 74.8 2.1 25.7 73.6 1.0 27.3 76.6 0.5
        34.9 68.9 -2.9 """
        table1 = np.fromstring(table1,sep=" ").reshape((12,5)).T
        Gmin = table1[0]
        Gmax = table1[1]
        #pick the appropriate omegaXYZ for the source’s magnitude:
        omegaX = table1[2][(Gmin<=G)&(Gmax>G)][0]
        omegaY = table1[3][(Gmin<=G)&(Gmax>G)][0] 
        omegaZ = table1[4][(Gmin<=G)&(Gmax>G)][0]
        pmraCorr = -1*sind(dec)*cosd(ra)*omegaX -sind(dec)*sind(ra)*omegaY + cosd(dec)*omegaZ
        pmdecCorr = sind(ra)*omegaX -cosd(ra)*omegaY
        return pmra-pmraCorr/1000., pmdec-pmdecCorr/1000.

    def PrepareConstraints(self, rv=False, catalog='gaiadr3.gaia_source', inflateFactor=1.):
        '''Retrieves parameters for both objects from Gaia EDR3 archive and computes system attriubtes,
        and assigns them to the Fitter object class.
        
        Args:
            rv (bool): flag for handling the presence or absence of RV measurements for both objects \
                in EDR3.  Gets set to True if both objects have Gaia RV measurements. Default = False
            catalog (str): name of Gaia catalog to query. Default = 'gaiaedr3.gaia_source'
            inflateFactor (flt): Factor by which to inflate the errors on Gaia proper motions to \
                account for improper uncertainty estimates.  Default = 1.0
        
        Written by Logan Pearce, 2020
        '''
        from astroquery.gaia import Gaia
        deg_to_mas = 3600000.
        mas_to_deg = 1./3600000.
        
        # Retrieve astrometric solution from Gaia EDR3
        job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(self.sourceid1))
        j = job.get_results()

        job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(self.sourceid2))
        k = job.get_results()

        if catalog == 'gaiadr2.gaia_source':
            # Retrieve RUWE from RUWE catalog for both sources and add to object state:
            job = Gaia.launch_job("SELECT * FROM gaiadr2.ruwe WHERE source_id = "+str(self.sourceid1))
            jruwe = job.get_results()

            job = Gaia.launch_job("SELECT * FROM gaiadr2.ruwe WHERE source_id = "+str(self.sourceid2))
            kruwe = job.get_results()

            self.ruwe1 = jruwe['ruwe'][0]
            self.ruwe2 = kruwe['ruwe'][0]
        else:
            # EDR3 contains ruwe in the main catalog:
            self.ruwe1 = j['ruwe'][0]
            self.ruwe2 = k['ruwe'][0]

        # Check RUWE for both objects and warn if too high:
        if self.ruwe1>1.2 or self.ruwe2>1.2:
            print('''WARNING: RUWE for one or more of your solutions is greater than 1.2. This indicates 
            that the source might be an unresolved binary or experiencing acceleration 
            during the observation.  Orbit fit results may not be trustworthy.''')

        # reference epoch:
        self.ref_epoch = j['ref_epoch'][0]

        # parallax:
        self.plx1 = [j[0]['parallax']*u.mas, j[0]['parallax_error']*u.mas]
        self.plx2 = [k[0]['parallax']*u.mas, k[0]['parallax_error']*u.mas]
        # RA/DEC
        self.RA1 = [j[0]['ra']*u.deg, j[0]['ra_error']*mas_to_deg*u.deg]
        self.RA2 = [k[0]['ra']*u.deg, k[0]['ra_error']*mas_to_deg*u.deg]
        self.Dec1 = [j[0]['dec']*u.deg, j[0]['dec_error']*mas_to_deg*u.deg]
        self.Dec2 = [k[0]['dec']*u.deg, k[0]['dec_error']*mas_to_deg*u.deg]
        # Proper motions
        pmRACorrected1,pmDecCorrected1 = self.edr3ToICRF(j[0]['pmra'],j[0]['pmdec'],j[0]['ra'],j[0]['dec'],j[0]["phot_g_mean_mag"])
        pmRACorrected2,pmDecCorrected2 = self.edr3ToICRF(k[0]['pmra'],k[0]['pmdec'],k[0]['ra'],k[0]['dec'],k[0]["phot_g_mean_mag"])
        self.pmRA1 = [pmRACorrected1*u.mas/u.yr, j[0]['pmra_error']*u.mas/u.yr*inflateFactor]
        self.pmRA2 = [pmRACorrected2*u.mas/u.yr, k[0]['pmra_error']*u.mas/u.yr*inflateFactor]
        self.pmDec1 = [pmDecCorrected1*u.mas/u.yr, j[0]['pmdec_error']*u.mas/u.yr*inflateFactor]
        self.pmDec2 = [pmDecCorrected2*u.mas/u.yr, k[0]['pmdec_error']*u.mas/u.yr*inflateFactor]
        # See if both objects have RV's in DR2:
        if catalog == 'gaiaedr3.gaia_source':
            key = 'dr2_radial_velocity'
            error_key = 'dr2_radial_velocity_error'
        elif catalog == 'gaiadr2.gaia_source':
            key = 'radial_velocity'
            error_key = 'radial_velocity_error'
        elif catalog == 'gaiadr3.gaia_source':
            key = 'radial_velocity'
            error_key = 'radial_velocity_error'
        if type(k[0][key]) == np.float64 and type(j[0][key]) == np.float64 or type(k[0][key]) == np.float32 and type(j[0][key]) == np.float32:
            rv = True
            self.rv1 = [j[0][key]*u.km/u.s,j[0][error_key]*u.km/u.s]
            self.rv2 = [k[0][key]*u.km/u.s,k[0][error_key]*u.km/u.s]
            rv1 = MonteCarloIt(self.rv1)
            rv2 = MonteCarloIt(self.rv2)
            self.rv = [ -np.mean(rv2-rv1) , np.std(rv2-rv1) ]   # km/s
            # negative to relfect change in coordinate system from RV measurements to lofti
            # pos RV = towards observer in this coord system
        else:
            self.rv = [0,0]

        # weighted mean of parallax values:
        plx = np.average([self.plx1[0].value,self.plx2[0].value], weights = [self.plx1[1].value,self.plx2[1].value])
        plxerr = np.max([self.plx1[1].value,self.plx2[1].value])
        self.plx = [plx,plxerr]                         # mas
        self.distance = distance(*self.plx)             # pc

        # Compute separations of component 2 relative to 1:
        r1 = MonteCarloIt(self.RA1)
        r2 = MonteCarloIt(self.RA2)
        d1 = MonteCarloIt(self.Dec1)
        d2 = MonteCarloIt(self.Dec2)
        ra = (r2*deg_to_mas - r1*deg_to_mas) * np.cos(np.radians(np.mean([self.Dec1[0].value,self.Dec2[0].value])))
        dec = ((d2 - d1)*u.deg).to(u.mas).value
        self.deltaRA = [np.mean(ra),np.std(ra)]         # mas
        self.deltaDec = [np.mean(dec),np.std(dec)]      # mas

        # compute relative proper motion:
        pr1 = MonteCarloIt(self.pmRA1)
        pr2 = MonteCarloIt(self.pmRA2)
        pd1 = MonteCarloIt(self.pmDec1)
        pd2 = MonteCarloIt(self.pmDec2)
        pmRA = [np.mean(pr2 - pr1), np.std(pr2-pr1)]    # mas/yr
        pmDec = [np.mean(pd2 - pd1), np.std(pd2 - pd1)] # mas/yr
        self.pmRA = masyr_to_kms(pmRA,self.plx)         # km/s
        self.pmDec = masyr_to_kms(pmDec,self.plx)       # km/s

        # Compute separation/position angle:
        r, p = to_polar(r1,r2,d1,d2)
        self.sep = tuple([np.mean(r).value, np.std(r).value])             # mas
        self.pa = tuple([np.mean(p).value, np.std(p).value])               # deg

        self.sep_au = tuple([((self.sep[0]/1000)*self.distance[0]), ((self.sep[1]/1000)*self.distance[0])])
        self.sep_km = tuple([ self.sep_au[0]*u.au.to(u.km) , self.sep_au[1]*u.au.to(u.km)])

        # compute total velocities:
        if rv:
            self.total_vel = [ add_in_quad([self.pmRA[0],self.pmDec[0],self.rv[0]]) , 
                            add_in_quad([self.pmRA[1],self.pmDec[1],self.rv[1]]) ]  # km/s
            self.total_planeofsky_vel = [ add_in_quad([self.pmRA[0],self.pmDec[0]]) , 
                            add_in_quad([self.pmRA[1],self.pmDec[1]]) ]             # km/s
        else:
            self.total_vel = [ add_in_quad([self.pmRA[0],self.pmDec[0]]) , 
                            add_in_quad([self.pmRA[1],self.pmDec[1]]) ]             # km/s
            self.total_planeofsky_vel = self.total_vel.copy()                       # km/s

        # compute deltamag:
        self.deltaGmag = j[0]['phot_g_mean_mag'] - k[0]['phot_g_mean_mag']
    

class FitOrbit(object):
    ''' Object for performing an orbit fit.  Takes attributes from Fitter class.

    ex: orbits = FitOrbit(fitterobject)

    Args:
        fitterobject (Fitter object): Fitter object initialized from the Fitter class
        write_stats (bool): If True, write out summary statistics of orbit sample at \
            conclusion of fit.  Default = True.
        write_results (bool):  If True, write out the fit results to a pickle file \
            in addition to the text file created during the fit.  Default = True.
        deltaRA, deltaDec (flt): relative separation in RA and Dec directions, in mas
        pmRA, pmDec (flt): relative proper motion in RA/Dec directions in km s^-1
        rv (flt, optional): relative RV of 2 relative to 1, if both are present in Gaia EDR3
        mtot_init (flt): initial total system mass in Msun from user input
        distance (flt): distance of system in pc, computed from Gaia parallax using method of Bailer-Jones et. al 2018.
        sep (flt): separation vector in mas
        pa (flt): postion angle of separation vector in degrees from North
        ref_epoch (flt): epoch of the measurement, 2016.0 for Gaia EDR3 and 2015.5 for Gaia DR2.
        Norbits (int): number of desired orbit samples
        write_stats (bool): if True, write summary of sample statistics to human-readable file at end of run.  Default = True
        write_results (bool): if True, write out current state of sample orbits in pickle file in periodic intervals during \
            run, and again at the end of the run.  RECOMMENDED.  Default = True
        results_filename (str): name of file for saving pickled results to disk.  If not supplied, \
            defaul name is FitResults.y.mo.d.h.m.s.pkl, saved in same directory as fit was run.
        stats_filename (str): name of file for saving human-readable file of stats of sample results. If not supplied, \
            defaul name is FitResults.Stats.y.mo.d.h.m.s.pkl, saved in same directory as fit was run.
        run_time (flt): run time for the last fit.  astropy units object


    Written by Logan Pearce, 2020

    '''
    def __init__(self, fitterobject, write_stats = True, write_results = True, python_version=False, \
                use_pm_cross_term = False, corr_coeff = None):
        # establish fit parameters:
        self.deltaRA = fitterobject.deltaRA
        self.deltaDec = fitterobject.deltaDec
        self.pmRA = fitterobject.pmRA
        self.pmDec = fitterobject.pmDec
        self.rv = fitterobject.rv
        self.mtot_init = fitterobject.mtot
        self.distance = fitterobject.distance
        self.sep = fitterobject.sep
        self.pa = fitterobject.pa
        self.ref_epoch = fitterobject.ref_epoch
        self.Norbits = fitterobject.Norbits
        self.write_results = write_results
        self.write_stats = write_stats
        self.results_filename = fitterobject.results_filename
        self.stats_filename = fitterobject.stats_filename
        self.astrometry = fitterobject.astrometry
        if self.astrometry:
            self.astrometric_ra = fitterobject.astrometric_ra
            self.astrometric_dec = fitterobject.astrometric_dec
            self.astrometric_dates = fitterobject.astrometric_dates
        self.use_user_rv = fitterobject.use_user_rv
        if self.use_user_rv:
            self.user_rv = fitterobject.user_rv
            self.user_rv_dates = fitterobject.user_rv_dates

        # run orbit fitter:
        self.fitorbit(python_fitOFTI=python_version, use_pm_cross_term = use_pm_cross_term, corr_coeff = corr_coeff)

    def fitorbit(self, save_results_every_X_loops = 100, python_fitOFTI=False, use_pm_cross_term = False, corr_coeff = None):
        '''Run the OFTI fitting run on the Fitter object.  Called when FitOrbit object
        is created.

        Args:
            save_results_every_X_loops (int): on every Xth loop, save status of the \
                orbit sample arrays to a pickle file, if write_results = True (Default)
            python_fitOFTI (bool): If True, fit using python only without using C Kepler's equation solver. Default = False
            use_pm_cross_term (bool): If True, include the proper motion correlation cross term in the Chi^2 computation \
                Default = False

        Written by Logan Pearce, 2020

        '''
        # write header:
        print('Saving orbits in',self.results_filename)
        k = open(self.results_filename, 'w')
        output_file_header = '# sma [arcsec]    period [yrs]    orbit phase    t_0 [yr]    ecc    incl [deg]\
        argp [deg]   lan [deg]    m_tot [Msun]    dist [pc]    chi^2    ln(prob)    ln(randn)'
        k.write(output_file_header + "\n")
        k.close()

        import time as tm
        ########### Perform initial run to get initial chi-squared: #############
        # Draw random orbits:
        #parameters = a,T,const,to,e,i,w,O,m1,dist
        numSamples = 10000
        parameters_init = draw_samples(numSamples, self.mtot_init, self.distance, self.ref_epoch)
        # Compute positions and velocities:
        if(python_fitOFTI):
                X,Y,Z,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,parameters=calc_OFTI(parameters_init,self.ref_epoch,self.sep,self.pa)
        else:
                returnArray = np.zeros((19,numSamples))
                returnArray = calcOFTI_C(parameters_init,self.ref_epoch,self.sep,self.pa,returnArray.copy())
                X,Y,Z,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot = returnArray[0:9]
                parameters = returnArray[9:]

        # Compute chi squared:
        if self.rv[0] != 0:
            model = np.array([Y,X,Ydot,Xdot,Zdot])
            data = np.array([self.deltaRA, self.deltaDec, self.pmRA, self.pmDec, self.rv])
        else:
            model = np.array([Y,X,Ydot,Xdot])
            data = np.array([self.deltaRA, self.deltaDec, self.pmRA, self.pmDec])
        chi2 = ComputeChi2(data,model)
        if use_pm_cross_term:
            chi2 -= ( 2 * corr_coeff * (data[2][0] - model[2]) * (data[3][0] - model[3]) ) / (data[2][1] * data[3][1]) 
            

        if self.astrometry:
            p = parameters.copy()
            a,T,const,to,e,i,w,O,m1,dist = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]
            chi2_astr = np.zeros(10000)
            # Calculate predicted positions at astr observation dates for each orbit:
            for j in range(self.astrometric_ra.shape[1]):
                # for each date, compute XYZ for each 10000 trial orbit.  We can
                # skip scale and rotate because that was accomplished in the calc_OFTI call above.
                X1,Y1,Z1,E1 = calc_XYZ(a,T,to,e,i,w,O,self.astrometric_dates[j])
                # Place astrometry into data array where: data[0][0]=ra obs, data[0][1]=ra err, etc:
                data = np.array([self.astrometric_ra[:,j], self.astrometric_dec[:,j]])
                # place corresponding predicited positions at that date for each trial orbit in arcsec:
                model = np.array([Y1*1000,X1*1000])
                # compute chi2 for trial orbits at that date and add to the total chi2 sum:
                chi2_astr += ComputeChi2(data,model)
            chi2 = chi2 + chi2_astr
        if self.use_user_rv:
            p = parameters.copy()
            a,T,const,to,e,i,w,O,m1,dist = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]
            chi2_rv = np.zeros(10000)
            for j in range(self.user_rv.shape[1]):
                # compute ecc anomaly at that date:
                X1,Y1,Z1,E1 = calc_XYZ(a,T,to,e,i,w,O,self.user_rv_dates[j])
                # compute velocities at that ecc anom:
                Xdot,Ydot,Zdot = calc_velocities(a,T,to,e,i,w,O,dist,E1)
                # compute chi2:
                chi2_rv += ComputeChi2(np.array([self.user_rv[:,j]]),np.array([Zdot]))
            chi2 = chi2 + chi2_rv
        print('inital chi min',np.nanmin(chi2))

        self.chi_min = np.nanmin(chi2)

        # Accept/reject:
        accepted, lnprob, lnrand = AcceptOrReject(chi2,self.chi_min)
        # count number accepted:
        number_orbits_accepted = np.size(accepted)
        # tack on chi2, log probability, log random unif number to parameters array:
        parameters = np.concatenate((parameters,chi2[None,:],lnprob[None,:],lnrand[None,:]), axis = 0)
        # transpose:
        parameters=np.transpose(parameters)
        # write results to file:
        k = open(self.results_filename, 'a')
        for params in parameters[accepted]:
            string = '   '.join([str(p) for p in params])
            k.write(string + "\n")
        k.close()
        
        ###### start loop ########
        # initialize:
        loop_count = 0
        start=tm.time()
        
        while number_orbits_accepted < self.Norbits:
            # Draw random orbits:
            numSamples = 10000
            parameters_init = draw_samples(numSamples, self.mtot_init, self.distance, self.ref_epoch)
            # Compute positions and velocities and new parameters array with scaled and rotated values:
            if(python_fitOFTI):
                X,Y,Z,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,parameters=calc_OFTI(parameters_init,self.ref_epoch,self.sep,self.pa)
            else:
                returnArray = np.zeros((19,numSamples))
                returnArray = calcOFTI_C(parameters_init,self.ref_epoch,self.sep,self.pa,returnArray.copy())
                X,Y,Z,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot = returnArray[0:9]
                parameters = returnArray[9:]
                returnArray = None
            # compute chi2 for orbits using Gaia observations:
            if self.rv[0] != 0:
                model = np.array([Y,X,Ydot,Xdot,Zdot])
                data = np.array([self.deltaRA, self.deltaDec, self.pmRA, self.pmDec, self.rv])
            else:
                model = np.array([Y,X,Ydot,Xdot])
                data = np.array([self.deltaRA, self.deltaDec, self.pmRA, self.pmDec])
            chi2 = ComputeChi2(data,model)
            if use_pm_cross_term:
                chi2 -= ( 2 * (data[2][0] - model[2]) * (data[3][0] - model[3]) ) / (data[2][1] * data[3][1]) 

            # add user astrometry if given:
            if self.astrometry:
                p = parameters.copy()
                a,T,const,to,e,i,w,O,m1,dist = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]
                chi2_astr = np.zeros(10000)
                # Calculate predicted positions at astr observation dates for each orbit:
                for j in range(self.astrometric_ra.shape[1]):
                    # for each date, compute XYZ for each 10000 trial orbit.  We can
                    # skip scale and rotate because that was accomplished in the calc_OFTI call above.
                    X1,Y1,Z1,E1 = calc_XYZ(a,T,to,e,i,w,O,self.astrometric_dates[j])
                    # Place astrometry into data array where: data[0][0]=ra obs, data[0][1]=ra err, etc:
                    data = np.array([self.astrometric_ra[:,j], self.astrometric_dec[:,j]])
                    # place corresponding predicited positions at that date for each trial orbit:
                    model = np.array([Y1*1000,X1*1000])
                    # compute chi2 for trial orbits at that date and add to the total chi2 sum:
                    chi2_astr += ComputeChi2(data,model)
                chi2 = chi2 + chi2_astr
            
            # add user rv if given:
            if self.use_user_rv:
                p = parameters.copy()
                a,T,const,to,e,i,w,O,m1,dist = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]
                chi2_rv = np.zeros(10000)
                for j in range(self.user_rv.shape[1]):
                    # compute ecc anomaly at that date:
                    X1,Y1,Z1,E1 = calc_XYZ(a,T,to,e,i,w,O,self.user_rv_dates[j])
                    # compute velocities at that ecc anom:
                    Xdot,Ydot,Zdot = calc_velocities(a,T,to,e,i,w,O,dist,E1)
                    # compute chi2:
                    chi2_rv += ComputeChi2(np.array([self.user_rv[:,j]]),np.array([Zdot]))
                chi2 = chi2 + chi2_rv
            
            # Accept/reject:
            accepted, lnprob, lnrand = AcceptOrReject(chi2,self.chi_min)
            
            if np.size(accepted) == 0:
                pass
            else:
                # count num accepted
                p = parameters.copy()
                a,T,const,to,e,i,w,O,m1,dist = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]
                sampleResults = calc_XYZ(a,T,to,e,i/180*np.pi,w/180*np.pi,O/180*np.pi,2016.0)
                number_orbits_accepted += np.size(accepted)
                parameters = np.concatenate((parameters,chi2[None,:],lnprob[None,:],lnrand[None,:]), axis = 0)
                parameters=np.transpose(parameters)
                

                k = open(self.results_filename, 'a')
                for params in parameters[accepted]:
                    string = '   '.join([str(p) for p in params])
                    k.write(string + "\n")
                k.close()

            if np.nanmin(chi2) < self.chi_min:
                # If there is a new min chi2:
                self.chi_min = np.nanmin(chi2)
                #print('found new chi min:',self.chi_min)
                # re-evaluate to accept/reject with new chi_min:
                
                if number_orbits_accepted != 0:
                    dat = np.genfromtxt(open(self.results_filename,"r"),delimiter='   ',ndmin=2)
                    lnprob = -(dat[:,10]-self.chi_min)/2.0
                    dat[:,11] = lnprob
                    accepted_retest = np.where(lnprob > dat[:,12])
                    q = open(self.results_filename, 'w')
                    q.write(output_file_header + "\n")
                    for data in dat[accepted_retest]:
                        string = '   '.join([str(d) for d in data])
                        q.write(string + "\n")
                    q.close()
                    dat2 = np.genfromtxt(open(self.results_filename,"r"),delimiter='   ',ndmin=2)
                    number_orbits_accepted=dat2.shape[0]
            
            loop_count += 1
            #print('loop count',loop_count)
            update_progress(number_orbits_accepted,self.Norbits)

        # one last accept/reject with final chi_min value:
        dat = np.genfromtxt(open(self.results_filename,"r"),delimiter='   ')
        lnprob = -(dat[:,10]-self.chi_min)/2.0
        dat[:,11] = lnprob
        accepted_retest = np.where(lnprob > dat[:,12])
        q = open(self.results_filename, 'w')
        q.write(output_file_header + "\n")
        for data in dat[accepted_retest]:
            string = '   '.join([str(d) for d in data])
            q.write(string + "\n")
        q.close()

        # when finished, upload results and store in object:
        dat = np.genfromtxt(open(self.results_filename,"r"),delimiter='   ')
        number_orbits_accepted=dat.shape[0]
        print('Final Norbits:', number_orbits_accepted)
        # intialise results object and store accepted orbits:
        if self.rv[0] != 0:
            self.results = Results(orbits = dat, limit_lan = False, limit_aop = False)
        else:
            self.results = Results(orbits = dat, limit_lan = False, limit_aop = False)
        self.results.Update(self.results.orbits)

        # pickle dump the results attribute:
        if self.write_results:
            self.results.SaveResults(self.results_filename.replace(".txt", ".pkl"), write_text_file = False)
        stop = tm.time()
        self.results.run_time = (stop - start)*u.s
        # compute stats and write to file:
        self.results.stats = Stats(orbits = self.results.orbits, write_to_file = self.write_stats, filename = self.stats_filename)
            
class Results(object):
    '''A class for storing and manipulating the results of the orbit fit.

    Args:
        orbits (Norbits x 13 array): array of accepted orbits from \
            OFTI fit in the same order as the following attributes
        sma (1 x Norbits array): semi-major axis in arcsec
        period (1 x Norbits array): period in years
        orbit_fraction (1 x Norbits array): fraction of orbit past periastron \
            passage the observation (2016) occured on.  Values: [0,1)
        t0 (1 x Norbits array): date of periastron passage in decimal years
        ecc (1 x Norbits array): eccentricity
        inc (1 x Norbits array): inclination relative to plane of the sky in deg
        aop (1 x Norbits array): arguement of periastron in deg
        lan (1 x Norbits array): longitude of ascending node in deg
        mtot (1 x Norbits array): total system mass in Msun
        distance (1 x Norbits array): distance to system in parsecs
        chi2 (1 x Norbits array): chi^2 value for the orbit
        lnprob (1 x Norbits array): log probability of orbit
        lnrand (1 x Norbits array): log of random "dice roll" for \
            orbit acceptance 
        limit_aop, limit_lan (bool): In the absence of radial velocity info, \
            there is a degeneracy between arg of periastron and long of ascending \
            node.  Common practice is to limit one to the interval [0,180] deg. \
            By default, lofti limits lan to this interval if rv = False.  The user can \
            choose to limit aop instead by setting limit_aop = True, limit_lan = False. \
            The orbits[:,6] (aop) and orbits[:,7] (lan) arrays preserve the original values. \

    Written by Logan Pearce, 2020

    '''
    def __init__(self, orbits = [], limit_aop = False, limit_lan = False):
        self.orbits = orbits
        self.limit_lan = limit_lan
        self.limit_aop = limit_aop

    def Update(self, orbits):
        '''Take elements of the "orbits" attribute and populate
        the orbital element attributes

        Args:
            orbits (arr): orbits array from Results class

        Written by Logan Pearce, 2020

        '''
        self.sma = orbits[:,0]
        self.period = orbits[:,1]
        self.orbit_fraction = orbits[:,2]
        self.t0 = orbits[:,3]
        self.ecc = orbits[:,4]
        self.inc = orbits[:,5]
        self.aop = orbits[:,6]
        if self.limit_aop:
            self.aop = limit_to_180deg(self.aop)
        self.lan = orbits[:,7] % 360
        if self.limit_lan:
            self.lan = limit_to_180deg(self.lan)
        self.mtot = orbits[:,8]
        self.distance = orbits[:,9]
        self.chi2 = orbits[:,10]
        self.lnprob = orbits[:,11]
        self.lnrand = orbits[:,12]

    def SaveResults(self, filename, write_text_file = False, text_filename = None):
        '''Save the orbits and orbital parameters attributes in a pickle file

        Args:
            filename (str): filename for pickle file
            write_text_file (bool): if True, also write out the accepted orbits to a \
                human readable text file
            text_filename (bool): if write_to_text = True, specifify filename for text file

        Written by Logan Pearce, 2020

        '''
        pickle.dump(self, open( filename, "wb" ) )

        # write results to file:
        if write_text_file:
            k = open(text_filename, 'a')
            for params in self.orbits:
                string = '   '.join([str(p) for p in params])
                k.write(string + "\n")
            k.close()

    def LoadResults(self, filename, append = False):
        '''Read in the orbits and orbital parameters attributes from a pickle file

        Args:
            filename (str): filename of pickle file to load
            append (bool): if True, append read in orbit samples to another Results \
                object.  Default = False.

        Written by Logan Pearce, 2020

        '''
        results_in = pickle.load( open( filename, "rb" ) )
        if append == False:
            self.orbits = results_in.orbits
            self.Update(self.orbits)
        else:
            self.orbits = np.vstack((self.orbits,results_in.orbits))
            self.Update(self.orbits)

    # plotting results:
    def PlotHists(self):
        '''Plot 1-d histograms of orbital elements 'sma','ecc','inc','aop','lan','t0' from fit results.

        Written by Logan Pearce, 2020

        '''
        if len(self.sma < 50):
            bins = 50
        else:
            bins = 'fd'
        fig = plt.figure(figsize=(30, 5.5))
        params = np.array([self.sma,self.ecc,self.inc,self.aop,self.lan,self.t0])
        names = np.array(['sma','ecc','inc','aop','lan','t0'])
        for i in range(len(params)):
            ax = plt.subplot2grid((1,len(params)), (0,i))
            plt.hist(params[i],bins=bins,edgecolor='none',alpha=0.8)
            plt.tick_params(axis='both', left=False, top=False, right=False, bottom=True, \
                    labelleft=False, labeltop=False, labelright=False, labelbottom=True)
            plt.xticks(rotation=45, fontsize = 20)
            plt.xlabel(names[i], fontsize = 25)
        plt.tight_layout()
        return fig

    def PlotOrbits(self, color = True, colorbar = True, ref_epoch = 2016.0, size = 100, \
            plot3d = False, cmap = 'viridis',xlim=False,ylim=False):
        '''Plot a random selection of orbits from the sample in the plane of the sky.

        Args:
            color (bool): if True, plot orbit tracks using a colormap scale to orbit fraction (phase) \
                past observation date (2015.5).  If False, orbit tracks will be black.  Default = True
            colorbar (bool): if True and color = True, plot colorbar for orbit phase
            ref_epoch (flt): reference epoch for drawing orbits.  Default = 2015.5 
            size (int): Number of orbits to plot.  Default = True
            plot3d (bool): If True, return a plot of orbits in 3D space. Default = False
            cmap (str): colormap for orbit phase plot

        Written by Logan Pearce, 2020

        '''
        # Random selection of orbits to plot:
        if len(self.sma) > size:
            # if there are more orbits than desired size, randomly select orbits from
            # the posterior sample:
            ind = np.random.choice(range(0,len(self.sma)),replace=False,size=size)
        else:
            # if there are fewer orbits than desired size, take all of them:
            ind = np.random.choice(range(0,len(self.sma)),replace=False,size=len(self.sma))

        from numpy import tan, arctan, sqrt, cos, sin, arccos
        # label for colormap axis:
        colorlabel = 'Phase'
        # create figure:
        fig = plt.figure(figsize = (7.5, 6.))
        plt.grid(ls=':')
        # invert X axis for RA:
        plt.gca().invert_xaxis()
        if plot3d:
            # Make 3d axis object:
            ax = fig.add_subplot(111, projection='3d')
            # plot central star:
            ax.scatter(0,0,0,color='orange',marker='*',s=300,zorder=10)
            ax.set_zlabel('Z (")',fontsize=20)
        else:
            # plot central star:
            plt.scatter(0,0,color='orange',marker='*',s=300,zorder=10)
        # For each orbit in the random selection from the posterior samples:
        for a,T,to,e,i,w,O in zip(self.sma[ind],self.period[ind],self.t0[ind],self.ecc[ind],np.radians(self.inc[ind]),\
                    np.radians(self.aop[ind]),np.radians(self.lan[ind])):
            # define an array of times along orbit:
            times = np.linspace(ref_epoch,ref_epoch+T,5000)
            X,Y,Z = np.array([]),np.array([]),np.array([])
            E = np.array([])
            # Compute X,Y,Z positions for each time:
            for t in times:
                n = (2*np.pi)/T
                M = n*(t-to)
                nextE = [danby_solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip([M],[e])]
                E = np.append(E,nextE)
            r1 = a*(1.-e*cos(E))
            f1 = sqrt(1.+e)*sin(E/2.)
            f2 = sqrt(1.-e)*cos(E/2.)
            f = 2.*np.arctan2(f1,f2)
            r = (a*(1.-e**2))/(1.+(e*cos(f)))
            X1 = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
            Y1 = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
            Z1 = r * sin(w+f) * sin(i)
            X,Y,Z = np.append(X,X1),np.append(Y,Y1),np.append(Z,Z1)
            # Plot the X,Y(Z) positions:
            if not plot3d:
                if color:
                    plt.scatter(Y,X,c=((times-ref_epoch)/T),cmap=cmap,s=3,lw=0)
                    plt.gca().set_aspect('equal', adjustable='datalim')
                else:
                    plt.plot(Y,X, color='black',alpha=0.3)
                    plt.gca().set_aspect('equal', adjustable='datalim')
            if plot3d:
                from mpl_toolkits.mplot3d import Axes3D
                if color:
                    ax.scatter(Y,X,Z,c=((times-ref_epoch)/T),cmap=cmap,s=3,lw=0)
                else:
                    ax.plot(Y,X,Z, color='black',alpha=0.3)
        # plot colorbar:
        if not plot3d:
            if color:
                if colorbar == True:
                    cb = plt.colorbar().set_label(colorlabel, fontsize=20)
                    plt.gca().tick_params(labelsize=14)

        plt.ylabel('Dec (")',fontsize=20)
        plt.xlabel('RA (")',fontsize=20)
        plt.gca().tick_params(labelsize=14)
        if(xlim):
            plt.xlim(xlim)
        if(ylim):
            plt.ylim(ylim)
        return fig

    def PlotSepPA(self, ref_epoch = 2016.0, size = 100, timespan = [20,20], orbitcolor = 'skyblue'):
        '''Plot a random selection of orbits from the sample in separation and position angle as 
        a function of time.

        Args:
            ref_epoch (flt): reference epoch for drawing orbits.  Default = 2015.5 
            size (int): Number of orbits to plot.  Default = True
            timespan (tuple, int): number of years before [0] and after [1] the ref epoch to \
                plot sep and pa
            orbitcolor (str): color to use to plot the orbits

        Written by Logan Pearce, 2020

        '''
        # Random selection of orbits to plot:
        if len(self.sma) > size:
            # if there are more orbits than desired size, randomly select orbits from
            # the posterior sample:
            ind = np.random.choice(range(0,len(self.sma)),replace=False,size=size)
        else:
            # if there are fewer orbits than desired size, take all of them:
            ind = np.random.choice(range(0,len(self.sma)),replace=False,size=len(self.sma))

        from numpy import tan, arctan, sqrt, cos, sin, arccos
        # make figure
        fig = plt.figure(figsize = (8, 10))
        # define subplots:
        plt.subplot(2,1,1)
        plt.gca().tick_params(labelsize=14)
        plt.grid(ls=':')
        # define times to compute sep/pa:
        tmin,tmax = ref_epoch - timespan[0],ref_epoch + timespan[1]
        t = np.linspace(tmin,tmax,2000)
        date_ticks = np.arange(tmin,tmax,10)
        # for each selected orbit from the sample:
        for a,T,to,e,i,w,O in zip(self.sma[ind],self.period[ind],self.t0[ind],self.ecc[ind],np.radians(self.inc[ind]),\
                    np.radians(self.aop[ind]),np.radians(self.lan[ind])):
            X = np.array([])
            Y = np.array([])
            # compute X,Y at each time point:
            X1,Y1 = orbits_for_plotting(a,T,to,e,i,w,O,t)
            X = np.append(X, X1)
            Y = np.append(Y,Y1)
            # compute sep:
            r=np.sqrt((X**2)+(Y**2))
            # plot sep in mas:
            plt.plot(t,r*1000,color=orbitcolor,alpha=0.5)
        plt.ylabel(r'$\rho$ (mas)',fontsize=20)

        # next suplot:
        plt.subplot(2,1,2)
        plt.grid(ls=':')
        # for each selected orbit from the sample:
        for a,T,to,e,i,w,O in zip(self.sma[ind],self.period[ind],self.t0[ind],self.ecc[ind],np.radians(self.inc[ind]),\
                    np.radians(self.aop[ind]),np.radians(self.lan[ind])):
            X = np.array([])
            Y = np.array([])
            X1,Y1 = orbits_for_plotting(a,T,to,e,i,w,O,t)
            X = np.append(X, X1)
            Y = np.append(Y,Y1)
            # compute pa:
            theta=np.arctan2(X,-Y)
            theta=(np.degrees(theta)+270.)%360
            # plot it:
            plt.plot(t,theta,color=orbitcolor,alpha=0.5)
        plt.ylabel(r'P.A. (deg)',fontsize=19)
        plt.xlabel('Years',fontsize=19)
        plt.gca().tick_params(labelsize=14)
        plt.tight_layout()
        return fig
            
class Stats(object):
    '''A class for storing and manipulating the statistics of the results of the orbit fit.

    For every parameter, there is a series of stats computed and saved as stats.param.stat

    Examples:
        stats.sma.mean = mean of semimajor axis
        stats.ecc.ci68 = 68% confidence interval for eccentricity
        stats.aop.std = standard deviation of arg of periastron

    Args:
        orbits (Norbits x 13 array): array of accepted orbits from \
            OFTI fit in the same order as the following attributes
        param.mean (flt): mean of parameter computed using np.mean
        param.median (flt): np.median of parameter
        param.mode (flt): mode of parameter
        param.std (flt): standard deviation from np.std
        param.ci68 (tuple,flt): 68% minimum credible interval of form (lower bound, upper bound)
        param.ci95 (tuple,flt): 95% minimum credible interval
        write_to_file (bool): If True, write stats to a human-readbale text file.
        filename (str): filename for saving stats file.  If not supplied, default \
            name is FitResults.Stats.y.mo.d.h.m.s.pkl, saved in same directory as fit was run.

        
    Written by Logan Pearce, 2020

    '''
    def __init__(self, orbits = [], write_to_file = False, filename = None):
        self.orbits = orbits
        # Compute stats on parameter arrays and save as attributes:
        self.sma = StatsSubclass(self.orbits[:,0])
        self.period = StatsSubclass(self.orbits[:,1])
        self.orbit_fraction = StatsSubclass(self.orbits[:,2])
        self.t0 = StatsSubclass(self.orbits[:,3])
        self.ecc = StatsSubclass(self.orbits[:,4])
        self.inc = StatsSubclass(self.orbits[:,5])
        self.aop = StatsSubclass(self.orbits[:,6])
        self.lan = StatsSubclass(self.orbits[:,7])
        self.mtot = StatsSubclass(self.orbits[:,8])
        self.distance = StatsSubclass(self.orbits[:,9])

        if write_to_file:
            params = np.array([self.sma,self.period,self.orbit_fraction,self.t0,self.ecc,self.inc,\
                self.aop,self.lan,self.mtot,self.distance])
            names = np.array(['sma','period','orbit fraction','t0','ecc','inc','aop','lan','mtot','distance'])
            if not filename:
                filename = 'FitResults.Stats.'+time.strftime("%Y.%m.%d.%H.%M.%S")+'.txt'
            k = open(filename, 'w')
            string = 'Parameter    Mean    Median    Mode    Std    68% Min Cred Int    95% Min Cred Int'
            k.write(string + "\n")
            for i in range(len(params)):
                string = make_parameter_string(params[i],names[i])
                k.write(string + "\n")
            k.close()

class StatsSubclass(Stats):
    '''Subclass for computing and storing statistics

    Args:
        array (arr): array for which to compute statistics
    '''
    def __init__(self, array):
        self.mean,self.median,self.mode,self.std,self.ci68,self.ci95 = compute_statistics(array)





        
        



