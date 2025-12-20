from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
        Read and parse a requirements file, returning a list of package dependencies.
        Reads a requirements.txt file line by line, strips whitespace from each line,
        and removes the '-e .' (editable install) entry if present. Returns a cleaned
        list of package requirements suitable for installation.
        :param file_path: Path to the requirements file to read
        :return: List of package requirements with whitespace stripped and '-e .' removed
    '''

    HYPEN_E_DOT = '-e .'
    requirements = []

    with open(file_path, 'r') as file:
        requirements = file.readlines()

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return [req.strip() for req in requirements if req.strip()]

setup(    
    name='ml_project',
    version='0.0.1',
    author='Dmitry Polischuk',
    author_email='dmitry.polischuk@gmail.com',
    packages=find_packages(), # type: ignore
    install_requires=get_requirements('requirements.txt')
)