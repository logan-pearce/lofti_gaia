import astropy.units as u
import numpy as np
from loftitools2 import *
# Astroquery throws some warnings we can ignore:
import warnings
warnings.filterwarnings("ignore")
'''
This module obtaines measurements from Gaia DR2 and runs through the LOFTI Gaia/OFTI 
wide stellar binary orbit fitting technique.
'''

class Fitter(object):
    '''
    Master object for all lofti functions and results.

    Attributes are tuples of (value,uncertainty) unless otherwise indicated.  Attributes
    with astropy units are retrieved from Gaia archive, attributes without units are
    computed from Gaia values.  All relative values are for object 2 relative to object 1.

    Attributes:
        sourceid1, sourceid2 (int): Gaia DR2 source IDs for the two objects to be fit.
            Fit will be performed for sourceid2 relative to sourceid1
        mass1, mass2 (flt): Tuple of (mass,unceertainty) for object 1 and object 2
            in solar masses
        Nsamples (int): Number of desired sample orbits.
        ruwe1, ruwe2 (flt): RUWE value from Gaia archive
        ref_epoch (flt): reference epoch in decimal years. For Gaia DR2 this is 2015.5
        plx1, plx2 (flt): parallax from Gaia DR2 in mas
        RA1, RA2 (flt): right ascension from Gaia DR2; RA in deg, uncertainty in mas
        Dec1, Dec2 (flt): declination from Gaia DR2; Dec in deg, uncertainty in mas
        pmRA1, pmRA2 (flt): proper motion in RA in mas yr^-1 from Gaia DR2
        pmDec1, pmDec2 (flt): proper motion in DEC in mas yr^-1 from Gaia DR2
        rv1, rv2 (flt, optional): radial velocity in km s^-1 from Gaia DR2
        rv (flt, optional): relative RV of 2 relative to 1, if both are present in Gaia DR2
        plx (flt): weighted mean parallax for the binary system in mas
        distance (flt): distance of system in pc, computed from Gaia parallax using method
            of Bailer-Jones et. al 2018.
        deltaRA, deltaDec (flt): relative separation in RA and Dec directions, in mas
        pmRA, pmDec (flt): relative proper motion in RA/Dec directions in km s^-1
        sep (flt): total separation vector in mas
        pa (flt): postion angle of separation vector in degrees from North
        sep_au (flt): separation in AU
        sep_km (flt): separation in km
        total_vel (flt): total velocity vector in km s^-1.  If RV is available for both,
            this is the 3d velocity vector; if not it is just the plane of sky velocity.
        total_planeofsky_velocity (flt): total velocity in the plane of sky in km s^-1. 
            In the absence of RV this is equivalent to the total velocity vector.
        deltaGmag (flt): relative contrast in Gaia G magnitude.  Does not include uncertainty.

    Written by Logan Pearce, 2020
    '''
    def __init__(self, sourceid1, sourceid2, mass1, mass2, Norbits = 100000):
        self.sourceid1 = sourceid1
        self.sourceid2 = sourceid2
        try:
            self.mass1 = mass1[0]
            self.mass1err = mass1[1]
            self.mass2 = mass2[0]
            self.mass2err = mass2[1]
            self.mtot = self.mass1 + self.mass2
            self.mtoterr = np.sqrt((self.mass1err**2) + (self.mass2err**2))
        except:
            raise ValueError('Masses must be tuples of (value,error), ex: mass1 = (1.0,0.05)')
        self.Norbits = Norbits
        # Get Gaia measurements, compute needed cosntraints, and add to object:
        self.PrepareConstraints()
        # Make an empty results object to store results:
        #self.results = Results(self.Norbits)

    def PrepareConstraints(self, rv=False):
        '''
        Retrieves parameters for both objects from Gaia DR2 archive and computes system attriubtes,
        and assigns them to the Fitter object class.
        
        Args:
            rv (bool): flag for handling the presence or absence of RV measurements for both objects
                in DR2.  Default = False
        
        Written by Logan Pearce, 2020
        '''
        from astroquery.gaia import Gaia
        deg_to_mas = 3600000.
        mas_to_deg = 1./3600000.
        
        # Retrieve astrometric solution from Gaia DR2
        job = Gaia.launch_job("SELECT * FROM gaiadr2.gaia_source WHERE source_id = "+str(self.sourceid1))
        j = job.get_results()

        job = Gaia.launch_job("SELECT * FROM gaiadr2.gaia_source WHERE source_id = "+str(self.sourceid2))
        k = job.get_results()

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
    def __init__(self, fitterobject):
        self.deltaRA = fitterobject.deltaRA
        self.deltaDec = fitterobject.deltaDec
        self.pmRA = fitterobject.pmRA
        self.pmDec = fitterobject.pmDec
        self.rv = fitterobject.rv
        self.mtot = fitterobject.mtot
        self.mtoterr = fitterobject.mtoterr
        self.distance = fitterobject.distance
        self.sep = fitterobject.sep
        self.pa = fitterobject.pa
        self.ref_epoch = fitterobject.ref_epoch
        self.Norbits = fitterobject.Norbits

        self.fitorbit()

    def fitorbit(self):
        import time as tm
        ########### Perform initial run to get initial chi-squared: #############
        # Draw random orbits:
        #parameters = a,T,const,to,e,i,w,O,m1,dist
        a,T,const,to,e,i,w,O,m1,dist = draw_samples(10000, [self.mtot, self.mtoterr], self.distance, self.ref_epoch)
             #a,T,const,to,e,i,w,O,d,m_tot,dist,rho,pa
            # Compute positions and velocities and new parameters array with scaled and rotated values:
        X,Y,Z,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2 = calc_OFTI(a,T,const,to,e,i,w,O,\
            self.ref_epoch,m1,dist,self.sep,self.pa)
        # Compute chi squared:
        if self.rv[0] != 0:
            measurements = np.array([Y,X,Ydot,Xdot,Zdot])
            model = np.array([self.deltaRA, self.deltaDec, self.pmRA, self.pmDec, self.rv])
        else:
            measurements = np.array([Y,X,Ydot,Xdot])
            model = np.array([self.deltaRA, self.deltaDec, self.pmRA, self.pmDec])
        chi2 = ComputeChi2(model,measurements)
        self.chi_min = np.nanmin(chi2)
        # print(self.chi_min)

        # Accept/reject:
        '''delta_chi = -(chi2-self.chi_min)/2.0
        A = np.exp(delta_chi)
        rand = np.random.uniform(0.0,1.0,10000) 
        accepted = np.where(A > rand)'''
        accepted, lnprob, lnrand = AcceptOrReject(chi2,self.chi_min)
        # count number accepted:
        number_orbits_accepted = np.size(accepted)
        parameters = np.zeros((13,10000))
        parameters[0,:],parameters[1,:],parameters[2,:],parameters[3,:],parameters[4,:],parameters[5,:], \
              parameters[6,:],parameters[7,:],parameters[8,:],parameters[9,:],parameters[10,:],parameters[11,:], \
              parameters[12,:] = a2,T2,const,to2,e,i,w,O2,m1,dist,chi2,lnprob,lnrand
        # tack on chi2, log probability, log random unif number to parameters array:
        #parameters = np.concatenate((parameters,chi2[None,:],lnprob[None,:],lnrand[None,:]), axis = 0)
        # transpose:
        parameters=np.transpose(parameters)
        # intialist results object and store accepted orbits:
        '''self.results = Results(self.Norbits, orbits = parameters[accepted])'''
        # Make file to store output:
        output_directory = '.'
        rank = 0
        output_file = output_directory + '/accepted_'+str(rank)
        k = open(output_file, 'w')
        k.write('# semimajoraxis[arcsec]    period[yrs]    orbitfrac    t_o[yr]    ecc    incl[deg]    argofperiastron[deg]    posangleofnodes[deg]\
            mass[Msun]   distance[pc]  chisquaredvalue    proboforbit    randnum' + "\n")
        k.close()
        
        ###### start loop ########
        # initialize:
        #number_orbits_accepted = 0
        loop_count = 0
        start=tm.time()
        
        while number_orbits_accepted < self.Norbits:
            # Draw random orbits:
            a,T,const,to,e,i,w,O,m1,dist = draw_samples(10000, [self.mtot, self.mtoterr], self.distance, self.ref_epoch)
             #a,T,const,to,e,i,w,O,d,m_tot,dist,rho,pa
            # Compute positions and velocities and new parameters array with scaled and rotated values:
            X,Y,Z,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2 = calc_OFTI(a,T,const,to,e,i,w,O,\
                self.ref_epoch,m1,dist,self.sep,self.pa)
            # compute chi2 for orbits:
            if self.rv[0] != 0:
                measurements = np.array([Y,X,Ydot,Xdot,Zdot])
                model = np.array([self.deltaRA, self.deltaDec, self.pmRA, self.pmDec, self.rv])
            else:
                measurements = np.array([Y,X,Ydot,Xdot])
                model = np.array([self.deltaRA, self.deltaDec, self.pmRA, self.pmDec])
            chi2 = ComputeChi2(model,measurements)
            
            # Accept/reject:
            accepted, lnprob, lnrand = AcceptOrReject(chi2,self.chi_min)

            parameters = np.zeros((13,10000))
            parameters[0,:],parameters[1,:],parameters[2,:],parameters[3,:],parameters[4,:],parameters[5,:], \
                parameters[6,:],parameters[7,:],parameters[8,:],parameters[9,:],parameters[10,:],parameters[1,:], \
                parameters[12,:] = a2,T2,const,to2,e,i,w,O2,m1,dist,chi2,lnprob,lnrand
            if np.size(accepted) == 0:
                #print('none accepted')
                pass
            else:
                # count num accepted
                number_orbits_accepted += np.size(accepted)
                parameters = np.zeros((13,10000))
                parameters[0,:],parameters[1,:],parameters[2,:],parameters[3,:],parameters[4,:],parameters[5,:], \
                    parameters[6,:],parameters[7,:],parameters[8,:],parameters[9,:],parameters[10,:],parameters[1,:], \
                    parameters[12,:] = a2,T2,const,to2,e,i,w,O2,m1,dist,chi2,lnprob,lnrand
                parameters=np.transpose(parameters)
                '''# Store results:
                self.results.orbits = np.vstack((self.results.orbits,parameters[accepted]))
                # update orbital parameters attributes:
                self.results.Update(self.results.orbits)
                #print('num orbits',number_orbits_accepted)'''
                # Write out to text file:
                k = open(output_file, 'a')
                for params in parameters[accepted]:
                    string = '   '.join([str(p) for p in params])
                    k.write(string + "\n")
                k.close()

            '''if np.nanmin(chi2) < self.chi_min:
                # If there is a new min chi2:
                self.chi_min = np.nanmin(chi2)
                # print('New chi min:',self.chi_min)
                # re-evaluate to accept/reject with new chi_min:
                accepted_recompute, lnprob, lnrand = AcceptOrReject(self.results.chi2,self.chi_min)
                self.results.orbits = self.results.orbits[accepted_recompute]
                self.results.Update(self.results.orbits)
                number_orbits_accepted = len(self.results.chi2)'''
            if np.nanmin(chi2) < self.chi_min:
                self.chi_min = np.nanmin(chi2)
                #print('Found new chi min: ',chi_min)
                found_new_chi_min = 'yes'
            else:
                found_new_chi_min = 'no'
            #if this minimum chi is less than the previously assigned chi_min, update the chi_min variable
            #to the new value, and write it out to this file. 
        
            if found_new_chi_min == 'yes' and number_orbits_accepted!=0: 
                ############## Recalculate old accepted orbits with new chi-min for acceptance #######
                dat = np.loadtxt(open(output_file,"rb"),delimiter='   ',ndmin=2)
                a,T,const,to,e,i,w,O,m1,dist,chi,lnprob,lnrand = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],\
                    dat[:,6],dat[:,7],dat[:,8],dat[:,9],dat[:,10],dat[:,11],dat[:,12]
                q = open(output_file, 'w')
                q.write('# semimajoraxis[arcsec]    period[yrs]    orbitfrac    t_o[yr]    ecc    incl[deg]    argofperiastron[deg]    posangleofnodes[deg]\
                        mass[Msun]   distance[pc]  chisquaredvalue    proboforbit    randnum' + "\n")
                acc = 0
                for a1,T1,const1,to1,e1,i1,w1,O1,m11,dist1,c1,A1,dice1  in zip(a,T,const,to,e,i,w,O,m1,dist,chi,lnprob,lnrand):
                    delta_chi1 = -(c1-self.chi_min)/2.0
                    AA = delta_chi1
                    if AA > dice1:
                        string = '   '.join([str(p) for p in [a1,T1,const1,to1,e1,i1,w1,O1,m11,dist1,c1,AA,dice1]])
                        q.write(string + "\n")
                    else:
                        pass
                q.close()
                dat2 = np.loadtxt(open(output_file,"rb"),delimiter='   ',ndmin=2)
                number_orbits_accepted=dat2.shape[0]
            else:
                pass

            if loop_count%10 == 0:
                dat2 = np.loadtxt(open(output_file,"rb"),delimiter='   ',ndmin=2)
                number_orbits_accepted=dat2.shape[0]
                update_progress(number_orbits_accepted,self.Norbits)

            
            loop_count += 1
            #print('loop count',loop_count)
            #update_progress(number_orbits_accepted,self.Norbits)

        stop = tm.time()
        print('Time',stop - start,((stop - start)*u.s).to(u.hr))

            


            
class Results(object):
    def __init__(self, Norbits, orbits = []):
        '''
        A class for storing the results of the orbit fit.

        Attributes:
            Norbits (int): number of requested samples
        
        Written by Logan Pearce, 2020
        '''
        self.orbits = orbits

    def Update(self, orbits):
        self.sma = orbits[:,0]
        self.period = orbits[:,1]
        self.orbit_fraction = orbits[:,2]
        self.t0 = orbits[:,3]
        self.ecc = orbits[:,4]
        self.inc = orbits[:,5]
        self.aop = orbits[:,6]
        self.lan = orbits[:,7]
        self.mtot = orbits[:,8]
        self.distance = orbits[:,9]
        self.chi2 = orbits[:,10]
        self.lnprob = orbits[:,11]
        self.lnrand = orbits[:,12]


        
        



