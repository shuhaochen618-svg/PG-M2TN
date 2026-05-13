from setuptools import setup, find_packages

setup(
    name="pg-m2tn",
    version="1.0.0",
    description="Physics-Guided Masked Multi-Task Network for Edge Battery Diagnostics",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Shuhao Chen",
    author_email="2023333541008@mails.zstu.edu.cn",
    url="https://github.com/shuhaochen618-svg/PG-M2TN",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "pandas>=1.5.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
