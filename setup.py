from setuptools import setup, find_packages

setup(
   name='iOpt',
   version='0.1',
   description='фреймворк для автоматического выбора значений параметров для математических моделей, ИИ и МО.',
   author='UNN Team',
   author_email='',
   python_requires='>=3.7',
   packages=find_packages(),
   install_requires=['numpy>=1.19',
                     'depq',
                     'cycler',
                     'kiwisolver',
                     'matplotlib>=3.3.2',
                     'scikit-learn',
                     'sphinx_rtd_theme',
                     'readthedocs-sphinx-search',
                     'sphinxcontrib-details-directive',
                     'autodocsumm'],
)
