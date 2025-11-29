# Installation Instructions

## Prerequisites
- Python 3.11 or higher
- System dependencies: `poppler`, `tesseract`

## Steps

0. **Install System Dependencies**
   Ensure you have `poppler` and `tesseract` installed on your system (e.g., via `conda`, `apt`, or `brew`).

# Installation Instructions

## Prerequisites
- Python 3.11 or higher
- System dependencies: `poppler`, `tesseract`

## Steps

0. **Install System Dependencies**
   Ensure you have `poppler` and `tesseract` installed on your system (e.g., via `conda`, `apt`, or `brew`).

1. **Install Poetry**
   If you haven't installed Poetry yet, run:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Create the Conda Environment**
   Create a new conda environment (replace `ingestion-env` with your desired name):
   ```bash
   conda create -n ingestion-env python=3.11
   ```

3. **Activate the Environment**
   Activate the newly created environment:
   ```bash
   conda activate ingestion-env
   ```

4. **Install Dependencies**
   Install the libraries defined in `pyproject.toml`:
   ```bash
   poetry install
   ```