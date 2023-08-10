from setuptools import setup

setup(
    name='starcat',
    version='0.0.0',
    packages=['starcat'],
    url='https://github.com/sarashenyy/starcat',
    license='MIT',
    author='Yueyue Shen',
    author_email='shenyy@bao.ac.cn',
    description='Star cluster analysis toolkit',
    package_dir={'starcat': 'starcat'},
    # include_package_data=True,
    package_data={'': ['LICENSE', 'README.md'],
                  'starcat': ['data/*']}
    # install_requires=['sphinx',
    #                   'numpy',
    #                   'scipy',
    #                   'matplotlib',
    #                   'astropy'],
)