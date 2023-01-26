"""Package setup."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements/base.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="vis4d",
    version="0.0",
    author="VIS @ ETH",
    author_email="i@yf.io",
    description="Vis4D Python package for Visual 4D scene understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://cv.ethz.ch/",
    project_urls={
        "Documentation": "https://github.com/syscv/",
        "Source": "https://github.com/syscv/",
        "Tracker": "https://github.com/syscv/",
    },
    packages=setuptools.find_packages(exclude=("tests", "tests.*", "docs.*")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=required,
    package_data={
        "vis4d": [
            "py.typed",
        ]
    },
    include_package_data=True,
)
