# Ticket Support Model

A text classifier that predicts which support team (DevOps, QA, Security, etc.) an IT support ticket should be routed to, based on its description. Uses a TF-IDF + Logistic Regression pipeline built with scikit-learn.

## Requirements

- Python 3.10+
- pip

## Setup

### 1. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy, run PowerShell as your user and allow local scripts once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Windows (cmd.exe):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Your shell prompt should now be prefixed with `(venv)`.

### 2. Install dependencies

With the virtual environment active:

```bash
pip install -r requirements.txt
```

This installs:

- `pandas` — reading/manipulating the ticket dataset
- `scikit-learn` — TF-IDF vectorizer, Logistic Regression, train/test split, metrics
- `matplotlib` / `seaborn` — confusion matrix plotting
- `joblib` — saving/loading the trained model

### 3. Configure data and model paths

`model.py` currently points to hardcoded paths:

```python
CSV_PATH   = r"D:\Projects\models\it_support_tickets.csv"
MODEL_PATH = r"D:\Projects\models\ticket_predict\ticket_model.joblib"
```

Update these to match your local setup, or point `CSV_PATH` at the included `it_support_tickets.csv`.

### 4. Run

```bash
python main.py
```

This trains the model on the CSV data, prints evaluation metrics (accuracy, precision, recall, F1), saves a confusion matrix plot, persists the trained pipeline to disk, and runs a few sample predictions.

## Deactivating the virtual environment

```bash
deactivate
```

## Presentation

See [`prezentare.txt`](prezentare.txt) for a presentation script covering data
exploration, preprocessing, and baseline model evaluation (Romanian, with an
English translation below it).