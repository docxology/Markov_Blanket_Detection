from setuptools import setup, find_packages

setup(
    name="dmbd",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "networkx>=2.6.0",
        "seaborn>=0.11.0",
    ],
    author="OpenManus Team",
    author_email="info@openmanus.org",
    description="Dynamic Markov Blanket Detection Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/openmanus/dmbd",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
) 