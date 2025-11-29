# Installation Instructions

## Prerequisites
- Python 3.11 or higher
- **CUDA 12.1**: This project strictly requires CUDA 12.1. We cannot ensure the code works with other CUDA versions.

## Steps

1. **Install Poetry**
   If you haven't installed Poetry yet, run:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Create the Conda Environment**
   Create a new conda environment (replace `inference-env` with your desired name):
   ```bash
   conda create -n inference-env python=3.11
   ```

3. **Activate the Environment**
   Activate the newly created environment:
   ```bash
   conda activate inference-env
   ```

4. **Install FAISS**
   Install `faiss-gpu` using Conda with the specific CUDA 12.1 version:
   ```bash
   conda install pytorch::faiss-gpu cuda-version=12.1 -c pytorch -c nvidia
   ```

5. **Install Dependencies**
   Install the remaining libraries defined in `pyproject.toml`:
   ```bash
   poetry install
   ```
