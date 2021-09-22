"""Package setup."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vist",
    version="0.0",
    author="VIS @ ETH",
    author_email="i@yf.io",
    description="VisT Python package for Visual Tracking",
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
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "Pillow",
        "plyfile",
        "psutil",
        "pycocotools",
        "pydantic",
        "pytoml",
        "PyYAML",
        "requests",
        "scalabel",
        "torch",
        "torchvision",
        "tqdm",
        "devtools",
    ],
    package_data={
        "vist": [
            "data/datasets/motchallenge.toml",
            "data/datasets/waymo.toml",
            "data/datasets/kitti.toml",
            "py.typed",
        ]
    },
    include_package_data=True,
)
