# RiftAnalysis: League of Legends Win Predictor

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

## About The Project

**RiftAnalysis** is a Deep Learning project designed to predict the outcome of League of Legends matches based on post-game statistics. By leveraging a neural network built with **PyTorch**, the model analyzes key performance indicators such as Kills, Gold Earned, Vision Score, and Objective control to classify the match result as a "Win" or "Loss".

The project implements a Multi-Layer Perceptron (MLP) architecture and serves as a study on tabular data classification, handling overfitting in small datasets, and the importance of feature engineering in competitive gaming analytics.

**Academic Context:**
This project was developed as the final outcome of a machine learning course on **Coursera**. The file `test.ipynb` serves as the graded assignment submission, containing the specific tasks and solution code required for course completion.

### Features
* **Custom Neural Network:** A feed-forward network with hidden layers, ReLU activation, and Dropout for regularization.
* **Data Preprocessing:** Automated scaling using `StandardScaler` to normalize game stats (Gold, CS, Damage).
* **Binary Classification:** Uses `BCELoss` (Binary Cross Entropy) and Sigmoid activation to output win probabilities.
* **Performance Metrics:** Generates Confusion Matrices, Precision/Recall reports, and Accuracy scores to evaluate the model.

## Tech Stack

* **Core Logic:** Python 3.12
* **Deep Learning Framework:** PyTorch
* **Data Manipulation:** Pandas
* **Preprocessing & Metrics:** Scikit-Learn

## Getting Started

To run this project locally, you need Python installed on your machine.

### Prerequisites

* Python 3.x
* pip (Python package manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/BBennemann/riftanalysis.git](https://github.com/BBennemann/riftanalysis.git)
    cd riftanalysis
    ```

2.  **Install dependencies:**
    You can install the required libraries directly using pip:
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

1.  **Load the Data:**
    Ensure your dataset is located in the `./data/` folder (e.g., `league_of_legends_data_large.csv`).

2.  **Run the Training Script:**
    Open the `main.ipynb` notebook to see the full training loop or `test.ipynb` to review the graded assignment tasks. The model will:
    * Load and clean the data.
    * Split into Training and Test sets.
    * Train the Neural Network for 1000 epochs.
    * Output the loss progress numerically in the console.

3.  **Evaluate the Results:**
    After training, the script will output the **Confusion Matrix** and the **Accuracy Score** on the test set.
    * *Example Output:*
        ```text
        Test Accuracy:  0.53
        [[ 0 95]
         [ 0 105]]
        ```
    * Use these metrics to adjust hyperparameters (Learning Rate, Hidden Layers) in `model.py`.

## Contributors

* **Bernardo Thomas Bennemann** - *Project Owner* - [BBennemann](https://github.com/BBennemann)
