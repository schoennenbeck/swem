from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pytorch-swem",
    version="0.2.3",
    author="Sebastian Schönnenbeck",
    author_email="schoennenbeck@gmail.com",
    url="https://github.com/schoennenbeck/swem",
    description="Pytorch implementation of the simple word embedding model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests"]),
    install_requires=["torch >= 1.7.1"],
    license="MIT",
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "pre-commit",
            "pytest",
            "pytest-cov",
            "coveralls",
            "coverage",
        ],
        "docs": [
            "sphinx",
            "myst-parser",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
        ],
    },
)
