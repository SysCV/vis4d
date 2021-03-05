"""Package setup."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="systm",
    version="0.1",
    author="CVL ETHZ",
    author_email="i@yf.io",
    description="SYSTM Python Package for perception and motion understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://cv.ethz.ch/",
    project_urls={
        "Documentation": "TODO provide link to documentation once established",
        "Source": "https://github.com/SysCV/systm",
        "Tracker": "https://github.com/SysCV/systm/issues",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "boto3",
        "dataclasses_json",
        "Flask",
        "Flask-Cors",
        "joblib",
        "numpy",
        "Pillow",
        "plyfile",
        "psutil",
        "pycocotools",
        "PyYAML",
        "requests",
        "tqdm",
    ],
)
