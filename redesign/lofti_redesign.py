import astropy.units as u
import numpy as np
from loftitools import *
# Astroquery throws some warnings we can ignore:
import warnings
warnings.filterwarnings("ignore")

class Fitter(object):
    def __init__(self, sourceid1, sourceid2, mass1, mass2):
        self.sourceid1 = sourceid1
        self.sourceid2 = sourceid2
        try:
            self.mass1 = mass1[0]
            self.mass1err = mass1[1]
            self.mass2 = mass2[0]
            self.mass2err = mass2[1]
        except:
            raise ValueError('Masses must be tuples of (value,error), ex: mass1 = (1.0,0.05)')

    def PrepareConstraints(self, rv=False):
        from astroquery.gaia import Gaia
        deg_to_mas = 3600000.
        mas_to_deg = 1./3600000.
        
        # Retrieve astrometric solution from Gaia DR2 and add to object state:
        job = Gaia.launch_job("SELECT * FROM gaiadr2.gaia_source WHERE source_id = "+str(self.sourceid1))
        j = job.get_results()

        job = Gaia.launch_job("SELECT * FROM gaiadr2.gaia_source WHERE source_id = "+str(self.sourceid2))
        k = job.get_results()

        # parallax:
        self.plx1 = [j[0]['parallax']*u.mas, j[0]['parallax_error']*u.mas]
        self.plx2 = [k[0]['parallax']*u.mas, k[0]['parallax_error']*u.mas]
        # RA/DEC
        self.RA1 = [j[0]['ra']*u.deg, j[0]['ra_error']*mas_to_deg*u.deg]
        self.RA2 = [k[0]['ra']*u.deg, k[0]['ra_error']*mas_to_deg*u.deg]
        self.Dec1 = [j[0]['dec']*u.deg, j[0]['dec_error']*mas_to_deg*u.deg]
        self.Dec2 = [k[0]['dec']*u.deg, k[0]['dec_error']*mas_to_deg*u.deg]
        # Proper motions:
        self.pmRA1 = [j[0]['pmra']*u.mas/u.yr, j[0]['pmra_error']*u.mas/u.yr]
        self.pmRA2 = [k[0]['pmra']*u.mas/u.yr, k[0]['pmra_error']*u.mas/u.yr]
        self.pmDec1 = [j[0]['pmdec']*u.mas/u.yr, j[0]['pmdec_error']*u.mas/u.yr]
        self.pmDec2 = [k[0]['pmdec']*u.mas/u.yr, k[0]['pmdec_error']*u.mas/u.yr]
        # See if both objects have RV's in DR2:
        if type(k[0]['radial_velocity']) == np.float64 and type(j[0]['radial_velocity']) == np.float64:
            rv = True
            self.rv1 = [j[0]['radial_velocity']*u.km/u.s,j[0]['radial_velocity_error']*u.km/u.s]
            self.rv2 = [k[0]['radial_velocity']*u.km/u.s,k[0]['radial_velocity_error']*u.km/u.s]
            rv1 = MonteCarloIt(self.rv1)
            rv2 = MonteCarloIt(self.rv2)
            self.rv = [ -np.mean(rv2-rv1) , np.std(rv2-rv1) ]   # km/s
            # negative to relfect change in coordinate system from RV measurements to lofti
            # pos RV = towards observer in this coord system
        else:
            self.rv = [0,0]
            
        # Retrieve RUWE for both sources and add to object state:
        job = Gaia.launch_job("SELECT * FROM gaiadr2.ruwe WHERE source_id = "+str(self.sourceid1))
        jruwe = job.get_results()

        job = Gaia.launch_job("SELECT * FROM gaiadr2.ruwe WHERE source_id = "+str(self.sourceid2))
        kruwe = job.get_results()
        self.ruwe1 = jruwe['ruwe'][0]
        self.ruwe2 = kruwe['ruwe'][0]
        if self.ruwe1>1.2 or self.ruwe2>1.2:
            yn = input('''WARNING: RUWE for one or more of your solutions is greater than 1.2. This indicates 
            that the source might be an unresolved binary or experiencing acceleration 
            during the observation.  Orbit fit results may not be trustworthy.  Do you 
            wish to continue?
            Hit enter to proceed, n to exit: ''')
            if yn == 'n':
                return None

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
        self.sep = [np.mean(r).value, np.std(r).value]              # mas
        self.pa = [np.mean(p).value, np.std(p).value]               # deg

        self.sep_au = [((self.sep[0]/1000)*self.distance[0]), ((self.sep[1]/1000)*self.distance[0])]
        self.sep_km = [ self.sep_au[0]*u.au.to(u.km) , self.sep_au[1]*u.au.to(u.km)]

        # compute total velocities:
        if rv:
            self.total_vel = [ add_in_quad([self.pmRA[0],self.pmDec[0],self.rv[0]]) , 
                            add_in_quad([self.pmRA[1],self.pmDec[1],self.rv[1]]) ]
            self.total_planeofsky_vel = [ add_in_quad([self.pmRA[0],self.pmDec[0]]) , 
                            add_in_quad([self.pmRA[1],self.pmDec[1]]) ]
        else:
            self.total_vel = [ add_in_quad([self.pmRA[0],self.pmDec[0]]) , 
                            add_in_quad([self.pmRA[1],self.pmDec[1]]) ]
            self.total_planeofsky_vel = self.total_vel.copy()

        # compute deltamag:
        self.deltaGmag = j[0]['phot_g_mean_mag'] - k[0]['phot_g_mean_mag']


    def fitorbit(self):
        try:
            test(self.deltaRA)
        except AttributeError:
            print('Error! You must run MyFitterObject.PrepareConstraints() before trying to fit an orbit')
            return

        



