from setuptools import setup, find_packages


setup(
    name="StandardGP",
    packages=find_packages(),
    version="0.1.0",
    license="BSD-3-Clause license",
    description="Regression algorithm",
    author="Thure Foken",
    author_email="thurefoken@gmail.com",
    url="https://github.com/thegenius89/StandardGP",
    download_url="https://github.com/thegenius89/StandardGP",
    keywords=["Regression", "ML", "GP", "Genetic Programming", "SGP"],
    python_requires=">=3.11.3"
    install_requires=[
        "numpy",
        "pmlb",
    ],
    classifiers=[
        "Development Status :: alpha unstable",
        "Intended Audience :: Researchers",
        "Programming Language :: Python :: 3",
    ],
)
