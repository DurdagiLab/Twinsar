from setuptools import setup, find_packages

setup(
    name="twinsar-8",
    version="1.0.0",
    description="Molecular twin detection using 8 single-element ratios",
    author="TwinSAR Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.4.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "rdkit>=2022.9.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "web": ["flask>=2.2.0"],
        "ml": ["joblib>=1.1.0", "catboost>=1.1.0", "optuna>=3.0.0"],
    },
    entry_points={
        "console_scripts": [
            "twinsar-8=app_8:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry :: Molecular Dynamics",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
