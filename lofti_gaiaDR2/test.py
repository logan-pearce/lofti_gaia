import numpy as np
# Gaia DR2 source ids:
DSTucA = 6387058411482257536
DSTucB = 6387058411482257280
# Mass estimates, must be a tuple of (value,uncertainty)
# in solar masses:
massA = (0.97, 0.04)
massB = (0.87, 0.04)

# Import the Fitter and FitOrbit objects:
from lofti import Fitter, FitOrbit
'''
## You can supply astrometry as a dictionary:
# in either sep/pa or relative ra/dec by setting column names:
astrometry = {'sep':np.array([5161.0, 5334.0, 5390.0]), 
            'seperr':np.array([299.5137, 299.5137, 299.5137]), 
            'pa':np.array([347.9, 347.61, 347.7]), 
            'paerr':np.array([0.294238, 0.294238, 0.294238]), 
            'dates':np.array([1998.516, 2009.7469, 2010.5])}

astrometry = {'ra':np.array([-1080.55, -1144.81, -1148.14,]), 
            'raerr':np.array([67.17, 70.12, 69.77]), 
            'dec':np.array([5048.98, 5205.51, 5264.79]), 
            'decerr':np.array([296.02, 288.52, 292.32]), 
            'dates':np.array([1998.516, 2009.7469, 2010.5]),
            }
# supply rv
rv = {'rv':np.array([-2.02, -1.67, -1.63, -1.92, -1.41]),
        'rverr':np.array([0.15, 0.53, 0.55, 0.35, 0.41,]),
        'dates':np.array([2007.39, 2018.87, 2018.88, 2018.88, 2018.88,])
        }

# Initialize the fitter object:
fitterobject = Fitter(DSTucA,           # source id object 1
                      DSTucB,           # source id object 2
                      massA,            # mass object 1
                      massB,            # mass object 2
                      Norbits = 50,     # number of desired accepted orbits for the posterior orbit sample
                      astrometry = astrometry,
                      user_rv = rv
                     )
print('Gaia rel RA/DEC:',fitterobject.deltaRA, fitterobject.deltaDec)
print('Astr rel RA/DEC:',fitterobject.astrometric_ra, fitterobject.astrometric_dec)
print('Gaia RV:',fitterobject.rv,'User RV:',fitterobject.user_rv)

## or as pandas dataframe with the correct column names:
import pandas as pd 
wds = pd.read_csv('docs/tutorials/WDS_DSTuc.csv', 
                    names = ['','sep','seperr','pa','paerr','dates'], # rename columns
                    header=0 # tell pandas that the first row is the header
                    )

# Initialize the fitter object:
fitterobject = Fitter(DSTucA,           # source id object 1
                      DSTucB,           # source id object 2
                      massA,            # mass object 1
                      massB,            # mass object 2
                      Norbits = 50,     # number of desired accepted orbits for the posterior orbit sample
                      astrometry = wds,
                      user_rv = rv
                     )
print('Gaia rel RA/DEC:',fitterobject.deltaRA, fitterobject.deltaDec)
print('Astr rel RA/DEC:',fitterobject.astrometric_ra, fitterobject.astrometric_dec)
print('Gaia RV:',fitterobject.rv,'User RV:',fitterobject.user_rv)
print()'''

###############################################################################

astrometry = {'ra':np.array([-1080.55, -1144.81, -1148.14,]), 
            'raerr':np.array([67.17, 70.12, 69.77]), 
            'dec':np.array([5048.98, 5205.51, 5264.79]), 
            'decerr':np.array([296.02, 288.52, 292.32]), 
            'dates':np.array([1998.516, 2009.7469, 2010.5]),
            }
# supply rv
rv = {'rv':np.array([-2.02, -1.67, -1.63, -1.92, -1.41]),
        'rverr':np.array([0.15, 0.53, 0.55, 0.35, 0.41,]),
        'dates':np.array([2007.39, 2018.87, 2018.88, 2018.88, 2018.88,])
        }

rv = {'rv':np.array([-1.67]),
        'rverr':np.array([0.53,]),
        'dates':np.array([2018.87,])
        }

# Initialize the fitter object:
fitterobject = Fitter(DSTucA,           # source id object 1
                      DSTucB,           # source id object 2
                      massA,            # mass object 1
                      massB,            # mass object 2
                      Norbits = 50,     # number of desired accepted orbits for the posterior orbit sample
                      astrometry = None,
                      user_rv = None
                     )
print('Gaia rel RA/DEC:',fitterobject.deltaRA, fitterobject.deltaDec)
#print('Astr rel RA/DEC:',fitterobject.astrometric_ra, fitterobject.astrometric_dec)
#print('Gaia RV:',fitterobject.rv,'User RV:',fitterobject.user_rv)
print()

# run orbit fit:
orbits = FitOrbit(fitterobject)
