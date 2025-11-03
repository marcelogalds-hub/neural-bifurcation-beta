"""
Setup do Neural Bifurcation Framework
"""

from setuptools import setup, find_packages

with open("LEIAME.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="neuralbifurcation",
    version="0.1.0-beta",
    author="Marcelo Galdino de Souza",
    author_email="marcelo.galdino@outlook.com.br",
    description="Framework multi-objetivo de early stopping para Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marcelogalds-hub/neural-bifurcation-beta",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.2.0",
    ],
    keywords="deep-learning machine-learning early-stopping multi-objective neural-networks",
    project_urls={
        "Bug Reports": "https://github.com/seu-usuario/neural-bifurcation/issues",  # ← TROCAR
        "Source": "https://github.com/seu-usuario/neural-bifurcation",  # ← TROCAR
    },
)
