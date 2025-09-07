"""
Setup script for EEG Bridge package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="eeg-bridge",
    version="1.0.0",
    author="AI Assistant (Claude)",
    description="Production-quality BCI processing for Unity integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "eeg-bridge=eeg_bridge.cli.main:main",
            "find-eeg-channels=eeg_bridge.utils.channel_finder:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
