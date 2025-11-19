from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name="nba_scraper",
    packages=find_packages(),
    include_package_data=True,
    version="1.2.2",
    license="GNU General Public License v3.0",
    description="Unified NBA play-by-play scraper for CDN and legacy sources",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matthew Barlowe",
    author_email="matt@barloweanalytics.com",
    url="https://github.com/mcbarlowe/nba_scraper",
    download_url="https://github.com/mcbarlowe/nba_scraper/archive/v1.2.2.tar.gz",
    keywords=["basketball", "NBA", "scraper"],
    install_requires=[
        "requests",
        "pandas",
        "numpy",
        "pyyaml",
        "nba_parser @ git+https://github.com/kmedved/nba_parser@master",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.7",
    ],
)
