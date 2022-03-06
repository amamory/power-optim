# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='power-optim',
    version='v0.1',
    description='Minizinc model of a power-aware task placement onto a heterogenous platform',
    long_description=readme,
    author='Alexandre Amory',
    author_email='amamory@gmail.com',
    url='https://github.com//amamory//power-optim',
    license=license,
    packages=find_packages(exclude=('docs'))
)


