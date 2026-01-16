"""Setup configuration for fibonacci-kv-cache package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read version from VERSION file
version_file = Path(__file__).parent / "VERSION"
version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else "0.0.0"

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text(encoding="utf-8").strip().split("\n")

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "hypothesis>=6.0.0",  # For property-based testing
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=0.990",
    "isort>=5.10.0",
]

setup(
    name="fibkvc",
    version=version,
    author="David Anderson",
    author_email="danders3@iit.edu",
    description="High-performance KV cache optimization using Fibonacci hashing for LLM deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/calivision/fibkvc",
    project_urls={
        "Bug Tracker": "https://github.com/calivision/fibkvc/issues",
        "Documentation": "https://fibkvc.readthedocs.io",
        "Source Code": "https://github.com/calivision/fibkvc",
        "Paper": "https://arxiv.org/abs/2601.xxxxx",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "benchmarks"]),
    package_data={
        "fibkvc": ["../VERSION"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    keywords="fibonacci hashing kv-cache llm diffusion-models optimization performance",
    include_package_data=True,
    zip_safe=False,
)
