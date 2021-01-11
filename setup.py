from setuptools import setup

def get_requires():
    reqs = []
    for line in open('requirements.txt', 'r').readlines():
        reqs.append(line)
    return reqs

setup(name='lofti_gaiaDR2',
      version='1.1.0',
      description='Orbit fitting with Gaia astrometry',
      url='https://github.com/logan-pearce/lofti_gaiaDR2',
      download_url='https://github.com/logan-pearce/lofti_gaiaDR2/archive/1.0.0.tar.gz',
      author='Logan Pearce',
      author_email='loganpearce1@email.arizona.edu',
      license='MIT',
      packages=['lofti_gaiaDR2'],
      zip_safe=False, 
      install_requires=get_requires()
          )
