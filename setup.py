"""Setup configuration for Single-Cell Graph Hub."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read version from __init__.py
version_file = Path(__file__).parent / "src" / "scgraph_hub" / "__init__.py"
version = "0.1.0"  # Default version
if version_file.exists():
    for line in version_file.read_text().splitlines():
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

# Core dependencies
core_requirements = [
    "torch>=2.0.0",
    "torch-geometric>=2.3.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "networkx>=2.6.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "tqdm>=4.62.0",
    "pyyaml>=6.0",
    "requests>=2.25.0",
    "h5py>=3.1.0",
    "anndata>=0.8.0",
    "scanpy>=1.8.0",
]

# Optional dependencies for full functionality
full_requirements = core_requirements + [
    "redis>=4.0.0",
    "aioredis>=2.0.0",
    "aiohttp>=3.8.0",
    "aiofiles>=0.8.0",
    "psutil>=5.8.0",
    "uvloop>=0.16.0",
    "numba>=0.56.0",
    "faiss-cpu>=1.7.0",
]

# Development dependencies
dev_requirements = full_requirements + [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.20.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
    "sphinx>=4.5.0",
    "sphinx-rtd-theme>=1.0.0",
    "jupyter>=1.0.0",
    "notebook>=6.4.0",
]

setup(
    name="single-cell-graph-hub",
    version=version,
    description="Graph Neural Networks for Single-Cell Omics Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Terragon Labs",
    author_email="contact@terragonlabs.com",
    url="https://github.com/terragonlabs/single-cell-graph-hub",
    project_urls={
        "Bug Reports": "https://github.com/terragonlabs/single-cell-graph-hub/issues",
        "Source": "https://github.com/terragonlabs/single-cell-graph-hub",
        "Documentation": "https://single-cell-graph-hub.readthedocs.io/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "scgraph_hub": [
            "data/catalogs/*.yaml",
            "data/configs/*.yaml",
            "data/schemas/*.json",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "full": full_requirements,
        "dev": dev_requirements,
        "redis": ["redis>=4.0.0", "aioredis>=2.0.0"],
        "async": ["aiohttp>=3.8.0", "aiofiles>=0.8.0", "uvloop>=0.16.0"],
        "performance": ["numba>=0.56.0", "faiss-cpu>=1.7.0"],
        "monitoring": ["psutil>=5.8.0"],
    },
    entry_points={
        "console_scripts": [
            "scgraph-hub=scgraph_hub.cli:main",
            "scgh=scgraph_hub.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "single-cell",
        "graph-neural-networks",
        "bioinformatics",
        "machine-learning",
        "genomics",
        "pytorch",
        "pytorch-geometric",
        "deep-learning",
        "omics",
        "cell-type-annotation",
    ],
    license="MIT",
    zip_safe=False,
    test_suite="tests",
    tests_require=["pytest>=7.0.0", "pytest-asyncio>=0.20.0"],
)