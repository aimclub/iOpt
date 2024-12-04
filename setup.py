from setuptools import setup, find_packages

from pathlib import Path
from typing import List

with open("README.md") as file:
    read_me_description = file.read()


def _readlines(*names: str, **kwargs) -> List[str]:
    encoding = kwargs.get('encoding', 'utf-8')
    lines = Path(__file__).parent.joinpath(*names).read_text(encoding=encoding).splitlines()
    return list(map(str.strip, lines))

def _extract_requirements(file_name: str):
    return [line for line in _readlines(file_name) if line and not line.startswith('#')]

def _get_requirements(req_name: str):
    requirements = _extract_requirements(req_name)
    return requirements

setup(
   name='iOpt',
   version='0.4.0',
   description='Framework for automatically tuning hyperparameter values for mathematical models, AI and ML.',
   author='UNN Team',
   author_email='',
   long_description=read_me_description,
   long_description_content_type="text/markdown",
   url="https://github.com/aimclub/iOpt",
   python_requires='>=3.9',
   packages=find_packages(exclude=["*test*", "examples", "benchmarks"]),
   install_requires=_get_requirements('requirements.txt'),
   classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
