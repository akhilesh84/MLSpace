# Environment Setup

This project requires Python 3.12 with specific packages. You can recreate the environment using one of the following methods:

## Option 1: Using the Conda environment file (recommended)

This will recreate the complete environment with all dependencies:

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate ml-env
```

For better cross-platform compatibility, you can use:

```bash
conda env create -f environment-no-builds.yml
```
