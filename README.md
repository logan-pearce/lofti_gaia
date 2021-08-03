# lofti_gaia
### Python package for orbit fitting with Gaia astrometry
Written by Logan A. Pearce, 2019<br>
If you use LOFTI in your work please cite [Pearce et al. (2020)](https://arxiv.org/abs/2003.11106).

## Complete documentation at [http://lofti-gaia.rtfd.io](http://lofti-gaia.rtfd.io)


### To install lofti_gaia:
    pip install lofti_gaia
   
### Required packages:
numpy, matplotlib, astropy, astroquery, pickle

Written for python 3.8

### Description:
lofti_gaia is a basic orbit fitter designed to fit orbital parameters for one wide stellar binary relative to the other, when both objects are resolved in Gaia.  It takes as input only the Gaia source id of the two components, and their masses.  It retrieves the relevant parameters from the Gaia archive, computes observational constraints for them, and fits orbital parameters to those measurements using a method based on Orbits for the Impatient (OFTI; Blunt et al. 2017).  It assumes the two components are bound in an elliptical orbit.  

Also included are some suggested basic statistics and plotting tools to examining the output from the fitter.

Caution:
 - It will give you answers, even if the two source ids you give it aren't actually bound.
 - It will give you answers even if the two Gaia astrometric solutions are not of good quality.
 
 Use with appropriate care and consideration of the reasonableness of your results!  If you give it garbage, it will give you garbage in return!
 
 For a detailed analysis of when using Gaia to constrain stellar binary orbits is and is not appropriate, see [Pearce et al. (2020)](https://arxiv.org/abs/2003.11106).

Please see the tutorial on the RTD page for how to use the package.

[![Documentation Status](https://readthedocs.org/projects/lofti-gaia/badge/?version=latest)](https://lofti-gaia.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/lofti-gaia.svg)](https://badge.fury.io/py/lofti-gaia)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3955152.svg)](https://doi.org/10.5281/zenodo.3955152)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![ASCL](https://img.shields.io/badge/ascl-2104.030-blue.svg?colorB=262255)](https://ascl.net/2104.030)

Copyright Logan Pearce, 2021

loganpearce1@email.arizona.edu
