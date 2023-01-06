from __future__ import print_function
import setuptools
from distutils.core import setup
from distutils.extension import Extension
import numpy as np
from distutils.ccompiler import new_compiler
import os
import sys
import tempfile
"""
Check for OpenMP based on
https://github.com/MDAnalysis/mdanalysis/tree/develop/package/setup.py
retrieved 06/15/15
"""
def get_requires():
    reqs = []
    for line in open('requirements.txt', 'r').readlines():
        reqs.append(line)
    return reqs
def hasfunction(cc, funcname, include=None, extra_postargs=None):
	# From http://stackoverflow.com/questions/
	#            7018879/disabling-output-when-compiling-with-distutils
	tmpdir = tempfile.mkdtemp(prefix='hasfunction-')
	devnull = oldstderr = None
	try:
		try:
			fname = os.path.join(tmpdir, 'funcname.c')
			f = open(fname, 'w')
			if include is not None:
				f.write('#include %s\n' % include)
			f.write('int main(void) {\n')
			f.write('    %s;\n' % funcname)
			f.write('}\n')
			f.close()
			# Redirect stderr to /dev/null to hide any error messages
			# from the compiler.
			# This will have to be changed if we ever have to check
			# for a function on Windows.
			devnull = open('/dev/null', 'w')
			oldstderr = os.dup(sys.stderr.fileno())
			os.dup2(devnull.fileno(), sys.stderr.fileno())
			objects = cc.compile([fname], output_dir=tmpdir, extra_postargs=extra_postargs)
			cc.link_executable(objects, os.path.join(tmpdir, "a.out"))
		except Exception as e:
			return False
		return True
	finally:
		if oldstderr is not None:
			os.dup2(oldstderr, sys.stderr.fileno())
		if devnull is not None:
			devnull.close()
"""
Check for OpenMP based on
https://github.com/MDAnalysis/mdanalysis/tree/develop/package/setup.py
"""
def detect_openmp():
	"""Does this compiler support OpenMP parallelization?"""
	compiler = new_compiler()
	print("Checking for OpenMP support... ")
	hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
	needs_gomp = hasopenmp
	if not hasopenmp:
		compiler.add_library('gomp')
	hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
	needs_gomp = hasopenmp
	if hasopenmp: print("Compiler supports OpenMP")
	else: print( "Did not detect OpenMP support.")
	return hasopenmp, needs_gomp
has_openmp, needs_gomp = detect_openmp()
parallel_args = ['-fopenmp', '-std=c99'] if has_openmp else ['-std=c99']
parallel_libraries = ['gomp'] if needs_gomp else []
def main():
    setup(name="lofti_gaia",
          version="2.0.5",
          description="Orbit fitting with Gaia astrometry",
          author="Logan Pearce",
		  url='https://github.com/logan-pearce/lofti_gaia',
          download_url='https://github.com/logan-pearce/lofti_gaia/archive/2.0.4.tar.gz',
		  packages =['lofti_gaia'],
		  zip_safe=False,
		  classifiers=[
			  "Development Status :: 5 - Production/Stable",
			  "Intended Audience :: Science/Research",
			  "License :: OSI Approved :: MIT License",
			  "Programming Language :: Python :: 3 :: Only",
			  "Programming Language :: C"
		  ],
          author_email="loganpearce1@arizona.edu",
		  licence="MIT",
          include_dirs = [np.get_include()],
          install_requires = get_requires(),
          ext_modules=[Extension("lofti_gaia.cFunctions", ["c_src/cFunctions.c"],extra_compile_args = parallel_args,libraries = parallel_libraries)])

if __name__ == "__main__":
    main()
