"""Setup script for FuncRoute"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="funcroute",
    version="0.1.0",
    author="FuncRoute Contributors",
    author_email="",
    description="Intelligent function/tool routing using FunctionGemma",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/funcroute",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "server": ["fastapi>=0.100.0", "uvicorn>=0.23.0"],
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0", "wordcloud>=1.9.0"],
    },
    entry_points={
        "console_scripts": [
            "funcroute=funcroute.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
