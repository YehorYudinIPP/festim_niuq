from setuptools import setup, find_packages

setup(
    name="festim_niuq",
    version="0.1.0",
    description="Non-Intrusive Uncertainty Quantification for FESTIM code",
    author="Yehor Yudin",
    author_email="y.yudinl@bangor.ac.uk",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib", 
        "pyyaml",
        "easyvvuq>=1.3",
        "chaospy>=4.3.13",
        "joblib",
        "emcee>=3.0",
        "arviz>=0.12",
        "corner>=2.2",
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'festim-uq=festim_niuq.uq.easyvvuq_festim:main',
        ],
    },
)
