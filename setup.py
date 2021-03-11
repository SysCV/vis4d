"""Package setup."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openmt",
    version="0.0",
    author="CVL @ ETHZ",
    author_email="i@yf.io",
    description="SYSTM Python Package for perception and motion understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://cv.ethz.ch/",
    project_urls={
        "Documentation": "https://github.com/syscv/",
        "Source": "https://github.com/syscv/",
        "Tracker": "https://github.com/syscv/",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
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
