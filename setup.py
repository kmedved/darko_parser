from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="darko_parser",  # <--- RENAMED from nba_scraper
    packages=find_packages(),
    include_package_data=True,
    version="1.2.2",
    license="GNU General Public License v3.0",
    description="Unified NBA play-by-play scraper and parser (Darko)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kostya Medvedovsky",
    url="https://github.com/kmedved/darko_parser",
    keywords=["basketball", "NBA", "scraper", "analytics"],
    install_requires=[
        "requests",
        "pandas",
        "numpy",
        "pyyaml",
        # "nba_parser" <-- REMOVED (It is now internal)
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.10",
    ],
)