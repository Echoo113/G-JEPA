from setuptools import setup, find_packages

setup(
    name="spark",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "tqdm",
        "matplotlib",
        "einops"
    ]
) 