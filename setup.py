from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = []
    # We need to strip comments and options since we might use pip-compile
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)

setup(
    name="swem",
    version="0.1.0",
    author="Sebastian Schönnenbeck",
    author_email="schoennenbeck@gmail.com",
    description="Pytorch implementation of the simple word embedding model.",
    packages=find_packages(exclude=["tests"]),
    install_requires=requires,
    license="LICENSE",
    python_requires=">=3.8",
    extra_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "pre-commit",
            "pytest",
        ]
    },
)
