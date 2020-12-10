from setuptools import setup, find_packages

setup(name='mixture_net',
      version='0.0.1',
      description='Template',
      author='Alberto Arrigoni',
      author_email='alberto.arrigoni@generali.com',
      url='https://github.com/aa-generali-italia/mixture_net',
      requires=['tensorflow', 'keras', 'numpy', 'pandas', 'scipy'],
      packages=find_packages()
     )
