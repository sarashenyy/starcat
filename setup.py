import glob
import os

from setuptools import setup, find_packages

data_files = [os.path.relpath(file, 'starcat') for file in glob.glob('starcat/data/**', recursive=True)]


with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='starcat',
    version='0.0.0',
    packages=find_packages(),
    url='https://github.com/sarashenyy/starcat',
    license='MIT',
    author='Yueyue Shen',
    author_email='shenyy@nao.cas.cn',
    description='Star cluster analysis toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'starcat': 'starcat'},
    # include_package_data=True,
    # package_data={'': ['LICENSE', 'README.md'],
    #               'starcat': ['data/*']},
    package_data={'': ['LICENSE', 'README.md'],
                  'starcat': data_files},
    install_requires=requirements
)