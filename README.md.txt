# SLA Breach Predictive Model

This project builds a **machine learning model** to predict whether a change request
will **breach its Service Level Agreement (SLA)**. It uses dummy change request data
to simulate a real IT service management (ITSM) environment.

The goal is to:
- Identify high-risk change requests **before** they breach SLAs
- Enable proactive intervention by IT / operations teams
- Demonstrate end-to-end data science workflow:
  - data exploration
  - feature engineering
  - model training
  - evaluation
  - model persistence

---

## 🔧 Tech Stack

- Python 3.x
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- matplotlib / seaborn (optional for EDA)
- joblib (or pickle) for model saving
- Azure ML (optional deployment pipeline)

---

## 📁 Project Structure

```text
SLAbreach/
├─ README.md                 # Project overview (this file)
├─ data/
│  └─ dummy_change_request_data.csv   # Sample dataset
├─ notebooks/
│  └─ SLA_Breach_Predictive_Model.ipynb   # Main analysis notebook
├─ src/
│  ├─ train_sla_breach_model.py      # Scripted training pipeline
│  └─ utils.py                       # (optional) helper functions
├─ models/
│  └─ gb_sla_breach_model.pkl        # Trained Gradient Boosting model
├─ requirements.txt                  # Python dependencies
└─ azure/
   └─ train_sla_breach_job.yml       # Azure ML job config
