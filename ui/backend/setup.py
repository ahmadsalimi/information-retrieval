import os

from setuptools import setup, find_packages

name = 'mir'
version = os.getenv('VERSION', 'local')
description = 'MIR gRPC service'
url = 'https://github.com/ahmadsalimi/information-retrieval'
author = 'Ahmad Salimi'
author_email = 'ahsa9978@gmail.com'


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name=name,
    version=version,
    description=description,

    url=url,
    author=author,
    author_email=author_email,

    packages=find_packages(),
    install_requires=requirements,

    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],

    entry_points={
        'console_scripts': [f'{name}={name}.cmd:main'],
    },
)
