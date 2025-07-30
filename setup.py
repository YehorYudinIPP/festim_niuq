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
        "easyvvuq",
        "chaospy",
        "joblib"
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'festim-uq=festim_niuq.uq.easyvvuq_festim:main',
        ],
    },
)
