from lofti_redesign import Fitter
import numpy as np

DSTucA = 6387058411482257536
DSTucB = 6387058411482257280
massA = (0.97, 0.04)
massB = (0.87, 0.04)

GL896A = 2824770686019003904
GL896B = 2824770686019004032

# Initialize the fitter object:
fitterobject = Fitter(GL896A,GL896B,massA,massB)
# Compute relevant properties for the system and add them 
# to the fitter object:
fitterobject.PrepareConstraints()
fitterobject.fitorbit()
# look at the system properties computed from Gaia:
for key,val in zip(fitterobject.__dict__.keys(),fitterobject.__dict__.values()):
    print(key,val)