from setuptools import setup, find_packages

setup(name='quiz-generator',
      version='0.0.1',
      description='',
      url='https://github.com/taminemelissa/quiz-generator',
      author='Mélissa Tamine, Adrien Servière',
      author_email='melissa.tamine@ensae.fr, adrien.serviere@ensae.fr',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      install_requires=['numpy', 'tqdm'])
