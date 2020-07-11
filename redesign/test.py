from lofti_redesign import Fitter, FitOrbit
import time as tm
#import numpy as np

DSTucA = 6387058411482257536
DSTucB = 6387058411482257280
massA = (0.97, 0.04)
massB = (0.87, 0.04)

GL896A = 2824770686019003904
GL896B = 2824770686019004032

# Initialize the fitter object:
#
fitterobject = Fitter(DSTucA,DSTucB,massA,massB,Norbits = 50)
start=tm.time()
orbits = FitOrbit(fitterobject)
stop = tm.time()
print('Time',stop - start,((stop - start)*u.s).to(u.min),((stop - start)*u.s).to(u.hr))
results = orbits.results

#fitterobject = Fitter(GL896A,GL896B,massA,massB,Norbits = 100)
#orbits = FitOrbit(fitterobject)

print(results.chi2)
print(len(results.chi2))
#print(fitterobject.results.lnprob)
#print(fitterobject.results.lnrand)
