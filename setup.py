from setuptools import setup, find_packages

setup(
    name="deepcropdx",
    version="1.0.0",
    description="Deep Learning Models for Plant Disease Detection",
    author="Jeremy Cleland",
    author_email="jeremy.cleland@icloud.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow>=2.7.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "pillow>=8.2.0",
        "jinja2>=3.0.0",
        "pyyaml>=6.0.0",
    ],
    entry_points={
        "console_scripts": [
            "deepcropdx-batch=src.main:main",
        ],
    },
    python_requires=">=3.8",
)
