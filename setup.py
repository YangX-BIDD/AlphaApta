from setuptools import setup, find_packages

setup(
    name="alphaapta",
    version="2.0.0",
    description="AlphaApta: a cell-conditioned computational framework for aptamer design",
    author="YangX-BIDD",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "einops>=0.7.0",
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "h5py>=3.10.0",
        "pyyaml>=6.0",
        "scanpy>=1.9.6",
        "anndata>=0.10.0",
        "mudata>=0.2.3",
        "muon>=0.1.5",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
)
