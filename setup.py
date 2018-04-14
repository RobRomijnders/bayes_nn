from setuptools import setup, find_packages

setup(name='bayes_nn',
      version='0.1',
      description='',
      url='',
      author='Rob_Romijnders',
      author_email='romijndersrob@gmail.com',
      license='MIT_license',
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'matplotlib',
          'torch',
          'python-mnist'
      ],
      packages=find_packages(exclude=('tests')),
      zip_safe=False)
