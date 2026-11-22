from setuptools import setup, find_packages

setup(
    name="alphaapta",
    version="1.0.0",
    description="AlphaApta: a cell-conditioned computational framework for aptamer design",
    author="YangX-BIDD",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "transformers>=4.35.0",
        "numpy",
        "pandas",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "scanpy",
        "muon",
        "matplotlib",
        "seaborn"
    ],
)
