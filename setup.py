#!/usr/bin/env python

import setuptools
import unittest

# Read the contents of the README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(name='fda',
    version='0.0.2',
    description='Package for doing a simple Fourier-based domain adaptation.',
    author='Luis Carlos Garcia Peraza Herrera',
    author_email='luiscarlos.gph@gmail.com',
    license='MIT',
    packages=[
        'fda',
    ],
    package_dir={
        'fda': 'src',
    },
    install_requires=[
      'numpy',
      'opencv-python',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
