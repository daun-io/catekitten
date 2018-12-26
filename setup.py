# Copyright 2018 Nyanye. All Rights Reserved.
# Author: nyanye (iam@nyanye.com)
from setuptools import find_packages
from setuptools import setup
import os
import io


# Dependencies without version specified
REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'sklearn',
    'fire',
]

# Retrieve version from about.py
def get_version():
    about = {}
    root = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(root, 'catekitten', 'about.py'), encoding='utf-8') as f:
        exec(f.read(), about)

    return about


def setup_package():
    about = get_version()
    setup(
        name='catekitten',
        version=about['__version__'],
        author=about['__author__'],
        author_email=about['__author_email__'],
        description='catekitten is a package for categorical classification experiment',
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages(),
        requires=[]
    )


if __name__ == '__main__':
    setup_package()