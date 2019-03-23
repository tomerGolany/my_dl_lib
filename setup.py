from setuptools import setup
from setuptools import find_packages

setup(name='my_dl_lib',
      version='0.1',
      description='Generic library to create neural networks and train them',
      url='http://github.com/tomergolany/',
      author='Tomer Golany',
      author_email='tomer.golany@gmail.com',
      license='Technion',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
