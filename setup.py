from setuptools import setup, find_packages
from os import path

version= {}

with open("mlhelpertools/__version__.py", "r") as f:
    exec(f.read(), version)

long_desc = None

with open(path.join(path.abspath(path.dirname(__file__)), "README.md"), "r") as f:
    long_desc = f.read()

setup(
    name="mlhelpertools",
    version=version["__version__"],
    description="A toolkit of commonly used ML tools/functions.",
    long_description=long_desc,
    url="https://github.com/AlexBarbera/ML-helper-tools",
    author="Alex Barbera",
    license="LICENSE.txt",
    packages=find_packages()
)