#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

# I used the following resources to compile the packaging boilerplate:
# https://python-packaging.readthedocs.io/en/latest/
# https://packaging.python.org/distributing/#requirements-for-packaging-and-distributing

from setuptools import find_packages, setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pytorch_lm',
      version='0.1',
      description='Pytorch-based experiments for Language Modeling',
      long_description=readme(),
      url='https://github.com/DavidNemeskey/pytorch_lm',
      author='Dávid Márk Nemeskey',
      license='MIT',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 1 - Just/Started',

          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          # This one is not in the list...
          'Topic :: Scientific/Engineering :: Natural Language Processing',

          # Environment
          'Operating System :: POSIX :: Linux',
          'Environment :: Console',
          'Natural Language :: English',

          # Pick your license as you wish (should match "license" above)
           'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
      keywords='language modeling RNN Pytoch',
      packages=find_packages(exclude=['scripts']),
      # Include the configuration -- unfortunately, MANIFEST.in doesn't seem
      # to do it for bdist (and package_data for sdist)
      # package_data={
      #     'conf': ['*'],
      # },
      # Install the scripts
      scripts=[
          'scripts/zaremba_pytorch.py',
          'scripts/zaremba_tf.py',
      ],
      # Tensorflow and numpy can be installed from requirement files, as they
      # are only required if the nn module / scripts are used.
      install_requires=[
          'numpy>=1.11.0',
          'tensorflow-gpu>=1.0',
          # Better regex
          'regex'
      ],
      # zip_safe=False,
      use_2to3=False)
