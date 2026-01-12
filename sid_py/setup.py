from setuptools import setup, find_packages

setup(
    name="sid_py",
    version="1.1",
    packages=find_packages(),
    install_requires=["numpy", "scipy"],
)