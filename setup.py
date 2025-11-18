from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README.md if it exists
this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="nba_analytics",
    version="0.1.0",
    description="Unified toolset for scraping NBA API data and calculating advanced stats (RAPM, Box Scores).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # This automatically finds 'nba_scraper' and 'nba_parser' in the root directory
    packages=find_packages(include=["nba_scraper*", "nba_parser*"]),
    include_package_data=True,
    
    author="Matthew Barlowe",
    author_email="matt@barloweanalytics.com",
    url="https://github.com/mcbarlowe/nba_analytics",  # Update if you push to a new remote
    license="GNU General Public License v3.0",
    
    python_requires=">=3.10",
    
    # Core dependencies (aligned with requirements.txt)
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "urllib3>=2.0.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.3.0",
        "SQLAlchemy>=2.0.0",
    ],
    
    # Optional dev dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "types-requests",
            "types-PyYAML",
        ]
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    
    keywords=["basketball", "NBA", "scraper", "analytics", "statistics"],
)