# Project: Automated Discovery of Parsimonious Functions for Complex Data via LLM-Guided Evolutionary Search

## 1. Abstract / Project Vision

This research-oriented project aims to develop a novel computational system for the automated discovery of the most parsimonious (i.e., shortest or simplest) mathematical functions that accurately represent complex, multi-dimensional data. The core of this endeavor lies in synergizing Large Language Models (LLMs) with evolutionary algorithms, particularly genetic programming, to navigate the vast search space of possible functions. A key challenge this project seeks to address is the current limitation of AI models in generating functions that perfectly match given graphical or numerical data while maintaining simplicity. The ultimate ambition extends to generating concise code representations (e.g., GLSL shaders) for efficient rendering or computation, applicable to data spanning 1D, 2D, 3D, and N-dimensions.

## 2. Problem Statement and Motivation

The automatic extraction of underlying mathematical laws or compact functional representations from observed data is a fundamental challenge in computer science, mathematics, and various scientific domains. While humans can often intuit simple functions from visual plots, automating this for complex, high-dimensional data, or noisy datasets is non-trivial. Current machine learning models, including LLMs, may generate plausible functional forms but often struggle with achieving perfect fidelity to the input data or ensuring true parsimony of the discovered function.

The motivations for this project are manifold:
*   **Enhanced Scientific Discovery:** Automating the derivation of concise models from experimental data can accelerate scientific insight.
*   **Efficient Data Representation:** Discovering compact functions serves as a powerful form of data compression.
*   **Optimized Computation:** Simple functions are often faster to evaluate, which is critical for applications like real-time graphics (e.g., procedural content generation, shader optimization).
*   **Advancing AI Capabilities:** Pushing the boundaries of how AI can "understand" and model complex data through symbolic reasoning.

## 3. Proposed Approach

This project proposes a multi-stage approach:

### 3.1. Input Data Preprocessing and Dimensionality Reduction

Given that raw complex data (e.g., images, 3D point clouds, N-D datasets) can be intractably high-dimensional, an initial preprocessing stage is envisioned. This may involve a dedicated component, conceptually termed **DimReduceNet**, to reduce the input data to a lower-dimensional, feature-rich representation that is more amenable to function search. For more details on this conceptual component, please see [`DimReduceNet.md`](./DimReduceNet.md).

### 3.2. LLM-Guided Evolutionary Search

The core of the function discovery process will employ an evolutionary algorithm, likely based on genetic programming. This algorithm will evolve populations of candidate functions to fit the (potentially reduced) input data. To make this search more intelligent and efficient, an LLM will be integrated to:

*   **Guide Evolution:** The LLM could learn to suggest promising initial function structures, mutations, or crossover operations.
*   **Parameterize Search:** Assist in setting appropriate parameters for the evolutionary algorithm based on data characteristics.
*   **Evaluate Fitness and Parsimony:** While the evolutionary algorithm will use explicit fitness functions (measuring accuracy and complexity), the LLM might provide nuanced evaluations or help in defining these fitness criteria. The definition of "parsimony" or "shortest function" is critical and will involve metrics such as symbolic expression length, tree complexity, or number of unique operators (see discussions in the exploration notebooks).

### 3.3. Specialized LLM Training Strategy

Drawing an analogy to Visual Language Models (VLMs), this project hypothesizes that training the LLM for this task might benefit from a specialized strategy. Instead of a generic multi-modal approach (e.g., directly mapping pixels to functions), the LLM might be trained more as a "symbolic reasoner" or a "code generator" operating on structured representations of data and mathematical expressions. This could be considered a "single-modal detour" focusing on the language of mathematics and symbolic manipulation, potentially leading to more precise and parsimonious function discovery.

## 4. Scope and Ambition

This project is inherently research-focused and ambitious. The initial stages will concentrate on 1D and simple 2D data, progressively moving towards higher dimensions and more complex function discovery. The exploration notebooks in this repository document the initial experiments with symbolic regression (e.g., using `gplearn`) and discussions on function complexity metrics.

The long-term vision includes:
*   Handling a wide variety of data types and dimensionalities.
*   Achieving high fidelity in matching functions to data.
*   Generating truly minimal functional representations.
*   Extending the system to output optimized code (e.g., GLSL) for specific applications.

---
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
