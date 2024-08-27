from setuptools import find_packages, setup

setup(
    name="ohw",
    version="0.1.0",
    packages=find_packages(include=['src']),
    package_dir={'' : 'src'}
)