from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="festim-niuq",
    version="0.1.0",
    description="Non-Intrusive Uncertainty Quantification for FESTIM code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yehor Yudin",
    author_email="y.yudin@bangor.ac.uk",
    url="https://github.com/YehorYudinIPP/festim_niuq",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pyyaml>=6.0",
        "easyvvuq>=1.3",
        "chaospy>=4.3.13",
        "joblib",
        "emcee>=3.0",
        "arviz>=0.12",
        "corner>=2.2",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
        "bayesian": [
            "emcee",
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "festim-uq=uq.easyvvuq_festim:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: OS Independent",
    ],
)
