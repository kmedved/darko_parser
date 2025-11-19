from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="darko_parser",
    packages=find_packages(),
    include_package_data=True,
    version="1.2.3",
    license="GNU General Public License v3.0",
    description="Unified NBA play-by-play scraper and parser (Darko)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kostya Medvedovsky",
    url="https://github.com/kmedved/darko_parser",
    keywords=["basketball", "NBA", "scraper", "analytics"],
    install_requires=[
        "requests>=2.31.0",
        "pandas>=2.2.0",
        "numpy>=1.26.4",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "black>=24.2.0",
            "codecov>=2.1.13",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.12",
    ],
)