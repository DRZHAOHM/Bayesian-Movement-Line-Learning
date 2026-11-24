# Bayesian Movement-Line Learning

Code and data for **Bayesian Movement-Line Learning for Joint Prediction of Spatio-Temporal Areal Data**. The repository contains lattice simulations and U.S. county unemployment forecasting experiments.

## Requirements

- **Hardware:** NVIDIA GeForce RTX 4060 or better recommended; 32 GB RAM recommended.
- **Software:** 64-bit Windows, Anaconda/Miniconda, **Python 3.13**, **PyTorch 2.15.0 (CUDA 13.2)**, NumPy, pandas, SciPy, NetworkX and Matplotlib.
- An up-to-date NVIDIA driver is required for GPU execution.

## Files

Keep all downloaded files in **one folder** (the repository root):

```text
README.md
R1_simulation.py          R2_simulation.py
R3_simulation.py          R4_simulation.py
US3108R1.py ... US3108R7.py
county_index.csv          county_month.csv
order_1.mtx               order_2.mtx ... order_4.mtx
counties_2024.geojson
```

- **Simulations:** `R1_simulation.py`–`R4_simulation.py` generate the simulated data internally; no external data files are required.
- **U.S. county application:** `county_index.csv`, `county_month.csv` and `order_1.mtx` are required. `county_month.csv` contains the prepared monthly county unemployment-rate panel (U.S. Bureau of Labor Statistics, Local Area Unemployment Statistics); `county_index.csv` specifies the county order; `order_1.mtx` contains first-order spatial adjacency. `order_2.mtx`–`order_4.mtx` are optional precomputed adjacency shells, validated if present. Higher orders are constructed from `order_1.mtx`.
- `counties_2024.geojson` provides county boundaries for mapping; it is **not required** to run the forecasting scripts.

## Installation (Anaconda Prompt)

Open **Anaconda Prompt** and run:

```bat
conda create -n movementline python=3.13 -y
conda activate movementline
python -m pip install numpy pandas scipy networkx matplotlib
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
cd /d "PATH_TO_REPOSITORY_FOLDER"
```

Replace `PATH_TO_REPOSITORY_FOLDER` with the folder containing the downloaded files. The CUDA check should print `True` for GPU execution. For other drivers or operating systems, follow the [official PyTorch installer](https://pytorch.org/get-started/locally/).

## Run

**Lattice simulations** (R = 1–4):

```bat
python -u R1_simulation.py
python -u R2_simulation.py
python -u R3_simulation.py
python -u R4_simulation.py
```

**U.S. county forecasts** (R = 1–7; run one at a time):

```bat
python -u US3108R1.py --device cuda --data-dir .
python -u US3108R2.py --device cuda --data-dir .
python -u US3108R3.py --device cuda --data-dir .
python -u US3108R4.py --device cuda --data-dir .
python -u US3108R5.py --device cuda --data-dir .
python -u US3108R6.py --device cuda --data-dir .
python -u US3108R7.py --device cuda --data-dir .
```

Each U.S. county script uses **its own R** for both movement-line modelling and predictive scoring (**no fixed-R4 scoring**). Results, prediction summaries and figures are written to script-specific output folders. Use `--resume` to continue an interrupted county run with unchanged settings.
