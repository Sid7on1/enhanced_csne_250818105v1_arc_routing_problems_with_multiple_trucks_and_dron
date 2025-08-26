import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from typing import List, Dict

# Define constants
PROJECT_NAME = "enhanced_cs.NE_2508.18105v1_Arc_Routing_Problems_with_Multiple_Trucks_and_Dron"
VERSION = "1.0.0"
DESCRIPTION = "Package for Arc Routing Problems with Multiple Trucks and Drones"
AUTHOR = "Abhay Sobhanana, Hadi Charkhgard, and Changhyun Kwon"
EMAIL = "author@example.com"
URL = "https://example.com"

# Define dependencies
INSTALL_REQUIRES: List[str] = [
    "torch",
    "numpy",
    "pandas",
]

# Define development dependencies
EXTRAS_REQUIRE: Dict[str, List[str]] = {
    "dev": [
        "pytest",
        "flake8",
        "mypy",
    ],
}

# Define package data
PACKAGE_DATA: Dict[str, List[str]] = {
    "": ["*.txt", "*.md"],
}

# Define entry points
ENTRY_POINTS: Dict[str, List[str]] = {
    "console_scripts": [
        "arc_routing=enhanced_cs.NE_2508.18105v1_Arc_Routing_Problems_with_Multiple_Trucks_and_Dron.main:main",
    ],
}

class CustomInstallCommand(install):
    """Custom install command to handle additional installation tasks."""
    def run(self):
        install.run(self)
        # Add custom installation tasks here

class CustomDevelopCommand(develop):
    """Custom develop command to handle additional development tasks."""
    def run(self):
        develop.run(self)
        # Add custom development tasks here

class CustomEggInfoCommand(egg_info):
    """Custom egg info command to handle additional egg info tasks."""
    def run(self):
        egg_info.run(self)
        # Add custom egg info tasks here

def main():
    """Main function to setup the package."""
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        packages=find_packages(),
        package_data=PACKAGE_DATA,
        entry_points=ENTRY_POINTS,
        cmdclass={
            "install": CustomInstallCommand,
            "develop": CustomDevelopCommand,
            "egg_info": CustomEggInfoCommand,
        },
    )

if __name__ == "__main__":
    main()