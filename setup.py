from distutils.core import setup
from os import path
this_directory = path.abspath(path.dirname(__file__))

setup(
      name='ApolloCraterDetectionTool',
      version='1.0',
      description='Detection Tool',
      author='ADS project Team Nene',
      url='https://github.com/ese-msc-2022/',
      packages=['apollo']
      )
