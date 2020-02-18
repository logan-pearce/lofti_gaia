# lofti_gaiaDR2
### Python package for orbit fitting with Gaia astrometry
Written by Logan A. Pearce, 2019<br>
If you use LOFTI in your work please cite Pearce et al. 2020 (submitted to ApJ)

### To install lofti_gaiaDR2:
    pip install lofti_gaiaDR2
   
### Required packages:
numpy, matplotlib, astropy, astroquery, pickle

Written for python 3.7 (I haven't tried it on python 2... may work?)

### Description:
lofti_gaia is a basic orbit fitter designed to fit orbital parameters for one wide stellar binary relative to the other, when both objects are resolved in Gaia DR2.  It takes as input only the Gaia DR2 source id of the two components, and their masses.  It retrieves the relevant parameters from the Gaia archive, computes observational constraints for them, and fits orbital parameters to those measurements using a method based on Orbits for the Impatient (OFTI; Blunt et al. 2017).  It assumes the two components are bound in an elliptical orbit.  

Also included are some suggested basic statistics and plotting tools to examining the output from the fitter.

Caution:
 - It will give you answers, even if the two source ids you give it aren't actually bound.
 - It will give you answers even if the two Gaia astrometric solutions are not of good quality.
 
 Use with appropriate care and consideration of the reasonableness of your results!  If you give it garbage, it will give you garbage in return!
 
 For a detailed analysis of when using Gaia to constrain stellar binary orbits is and is not appropriate, see Pearce et al. (2019) (currently in prep, email corresponding author for a copy)

Please see the tutorial for how to use the package.

Copyright Logan Pearce, 2019

loganpearce1@email.arizona.edu
