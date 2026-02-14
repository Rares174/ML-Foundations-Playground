# ML Foundations Playground

**An interactive machine learning playground for beginners and enthusiasts to experiment with foundational ML concepts.**

---

## Overview

ML Foundations Playground is a web-based application that allows users to explore core machine learning algorithms, visualize datasets, and understand how features impact predictions. Designed as a learning tool rather than a production ML platform, it guides users step-by-step through ML workflows in an interactive, visual way.

Key features include:

* Supervised learning (Linear Regression, Logistic Regression, KNN, Decision Trees, Random Forest)
* Unsupervised learning (KMeans clustering)
* Interactive feature selection and target specification
* Built-in demo datasets (Iris, Titanic, House Prices)
* Data analysis and visualization (missing values, feature distributions, correlations)
* Real-time ML training plots to illustrate algorithm behavior
* Simple vs advanced modes to cater to beginners and more advanced users
* Explanations of model metrics and outputs to reinforce learning

---

## Demo

The app provides preloaded demo datasets to experiment with:

* **Iris Dataset** – classic classification problem
* **Titanic Dataset** – survival prediction example
* **House Prices Dataset** – regression example

Users can also upload their own CSV datasets for training models or clustering analysis.

---

## Installation

### Prerequisites

* Docker & Docker Compose installed (recommended for cross-platform compatibility)
* Python 3.10+ if running without Docker

---

### Using Docker (recommended)

1. Clone the repository:

```bash
git clone <your-repo-url>
cd ML-Foundations-Playground
```

2. Build and run the Docker container:

```bash
docker compose up --build
```

3. Open your browser and visit:

```
http://localhost:5000
```

---

### Running Locally Without Docker

1. Clone the repository:

```bash
git clone <your-repo-url>
cd ML-Foundations-Playground
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

4. Run the backend server:

```bash
python backend/app.py
```

5. Open your browser at:

```
http://localhost:5000
```

---

## Usage

1. **Select a dataset**: choose a demo dataset or upload your own CSV.
2. **Feature selection**: pick which columns to use as features and optionally select a target column.
3. **Choose a model**: supervised algorithms require a target column; KMeans works without one.
4. **Train the model**: view plots, metrics, and step-by-step explanations of the training process.
5. **Analyze results**: understand the contribution of each feature and explore model outputs in real-time.

---

## Tech Stack

* Python (Flask) – backend API
* JavaScript – frontend interface
* Pandas, NumPy, scikit-learn – data manipulation and ML
* Matplotlib – visualization of features and model behavior
* Docker – containerized deployment

---

## Contributing

Contributions are welcome!

* Please open issues for bugs or feature requests.
* For code contributions, fork the repo, create a branch, and submit a pull request.

---

## License

[MIT License](LICENSE)
