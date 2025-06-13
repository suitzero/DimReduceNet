# Project: Shortest Function Finder

This project aims to explore methods for finding the shortest mathematical function to represent given data (1D, 2D, 3D graphs) and eventually generate concise code (e.g., GLSL) for rendering scenes.

## Explorations and Experiments

The `notebooks/` directory contains Jupyter Notebooks used for experimentation and exploration of different techniques for this project.

### Current Notebooks:

*   **`01_initial_exploration.ipynb`**:
    *   Demonstrates generation of sample 1D datasets.
    *   Includes experiments with symbolic regression using the `gplearn` library to automatically discover mathematical functions that fit the generated data.
    *   Shows examples of tuning `gplearn` parameters (`population_size`, `generations`, `parsimony_coefficient`, etc.) and observing their impact on the resulting functions and accuracy.
    *   Tests the `gplearn` approach on different types of 1D datasets to evaluate its flexibility.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd shortest_function_finder
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter Notebook or JupyterLab:**
    ```bash
    jupyter lab  # or jupyter notebook
    ```
    Then navigate to the `notebooks/` directory to open and run `01_initial_exploration.ipynb`.
