# This file is if we want to be able to pip install it & upload the package to pypi

from setuptools import setup


setup(name='chicago_preprocessor',
      version='0.0.1',
      install_requires=[
      'numpy',
      'pandas',
      'sklearn'
      ]
      description='Example preprocessor sklearn pipeline',
      author='Mike Swirsky',
      author_email='mike.swirsky@gmail.com',
      url='https://github.com/mswirsky/preprocessing_pipeline_chicago',
      packages=['chicago_preprocessor'],
      license='MIT',
      )
