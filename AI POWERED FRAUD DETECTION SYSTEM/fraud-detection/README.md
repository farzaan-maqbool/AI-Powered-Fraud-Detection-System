# 🛡️ AI-Powered Fraud Detection System

An advanced machine learning system for detecting fraudulent transactions in real-time, built with Python and Streamlit.

---

## 🚀 Features

- **Multiple ML Models** — Random Forest, Gradient Boosting, Logistic Regression, XGBoost (GBM)
- **Anomaly Detection** — Isolation Forest layer running alongside the classifier
- **Real-time Prediction** — Analyze individual transactions with risk factor breakdown
- **Batch Scoring** — Upload a CSV and score thousands of transactions at once
- **Live Simulation** — Animated transaction stream with live fraud probability feed
- **Model Analytics** — ROC curve, PR curve, confusion matrix, feature importances, model comparison
- **Alert Log** — Session history of all analyzed transactions, exportable to CSV
- **13 Features** — Rich feature set covering behavioral, geographic, and device signals

---

## 📁 Project Structure

```
fraud-detection/
├── app.py               # Streamlit frontend (6 tabs)
├── model.py             # ML training, inference, batch predict
├── data_generator.py    # Synthetic transaction data generator
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Installation

**1. Clone or download the project**

```bash
git clone <your-repo-url>
cd fraud-detection
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
streamlit run app.py
```

---

## 🧠 ML Features Used

| Feature | Description |
|---|---|
| `amount` | Transaction amount in USD |
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day of week (0=Mon, 6=Sun) |
| `distance_from_home_km` | Distance from user's home location |
| `is_foreign_transaction` | Whether transaction is outside home country |
| `is_high_risk_country` | Country flagged as high risk (NG, RU, CN) |
| `num_transactions_last_hour` | Transaction velocity — last hour |
| `num_transactions_last_day` | Transaction velocity — last day |
| `ip_risk_score` | IP address risk score (0–1) |
| `is_new_device` | Whether device is unrecognized |
| `account_age_days` | Age of the user account |
| `failed_attempts_last_hour` | Failed login/payment attempts |
| `amount_vs_user_avg` | Ratio of transaction amount to user's average |
| `merchant_category` | Category of merchant (grocery, travel, etc.) |
| `card_type` | Credit, debit, or prepaid |
| `device_type` | Mobile, desktop, tablet, POS terminal |
| `country` | Transaction country |

---

## 📊 App Tabs

| Tab | Description |
|---|---|
| 📊 Dashboard | KPIs, fraud trends by hour/category/country, correlation heatmap |
| 🔎 Predict | Single transaction analysis with risk factor breakdown |
| 📁 Batch Upload | Score a full CSV file, download results |
| 🔴 Live Simulation | Animated real-time transaction stream |
| 📈 Model Analytics | ROC, PR curve, confusion matrix, feature importances, model comparison |
| 🚨 Alert Log | History of all predictions made in the session |

---

## 🔧 Configuration (Sidebar)

- **Training samples** — 1,000 to 10,000 synthetic transactions
- **ML Model** — choose from 4 algorithms
- **Fraud threshold** — adjust the probability cutoff for flagging fraud (default 0.5)

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Frontend UI |
| `scikit-learn` | ML models, metrics, preprocessing |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Charts and visualizations |
| `seaborn` | Heatmaps and styled plots |

---

## 📌 Notes

- The system uses **synthetic data** generated with realistic fraud patterns. For production use, replace `data_generator.py` with your real transaction dataset.
- Trained model artifacts are saved to `model_artifacts.pkl` in the project directory and reused across sessions.
- The fraud threshold is adjustable — lower it to catch more fraud (higher false positives), raise it to reduce false alarms.

---

## 📄 License

MIT License — free to use and modify.
