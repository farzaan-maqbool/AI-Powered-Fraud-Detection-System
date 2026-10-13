import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time
import io
from datetime import datetime, timedelta
from data_generator import generate_transactions, MERCHANT_CATEGORIES, CARD_TYPES, DEVICE_TYPES, COUNTRIES
from model import train_model, load_model, predict_transaction, batch_predict

st.set_page_config(page_title="AI Fraud Detection", page_icon="🛡️", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
.fraud-badge  { background:#ff4b4b; color:white; padding:4px 10px; border-radius:8px; font-weight:700; }
.safe-badge   { background:#21c354; color:white; padding:4px 10px; border-radius:8px; font-weight:700; }
.warn-badge   { background:#ffa500; color:white; padding:4px 10px; border-radius:8px; font-weight:700; }
</style>""", unsafe_allow_html=True)

st.title("🛡️ AI-Powered Fraud Detection System")
st.caption("Advanced ML fraud detection with explainability, anomaly detection & real-time simulation")

# ── Session state init ───────────────────────────────────────────────────────
for key in ["model","scaler","encoders","iso","df","metrics","alert_log"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.alert_log is None:
    st.session_state.alert_log = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    n_samples   = st.slider("Training samples", 1000, 10000, 3000, 500)
    model_type  = st.selectbox("ML Model", ["Random Forest","Gradient Boosting","Logistic Regression","XGBoost (GBM)"])
    fraud_threshold = st.slider("Fraud threshold", 0.1, 0.9, 0.5, 0.05,
                                help="Probability above this = flagged as fraud")
    st.divider()
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Generating data & training…"):
            df = generate_transactions(n=n_samples)
            clf, scaler, encoders, metrics, X_test, y_test = train_model(df, model_type)
            st.session_state.update({"model":clf,"scaler":scaler,"encoders":encoders,
                                     "df":df,"metrics":metrics})
        st.success(f"✅ Trained on {n_samples} transactions")
        st.rerun()

    if st.session_state.model:
        st.divider()
        st.success("Model ready")
        m = st.session_state.metrics
        st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        st.metric("Avg Precision", f"{m['avg_precision']:.4f}")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "🔎 Predict", "📁 Batch Upload",
    "🔴 Live Simulation", "📈 Model Analytics", "🚨 Alert Log"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if st.session_state.df is None:
        st.info("Train a model using the sidebar to get started.")
    else:
        df = st.session_state.df
        fraud = df[df["is_fraud"] == 1]
        legit = df[df["is_fraud"] == 0]
        total = len(df)
        fraud_rate = len(fraud) / total

        # KPI row
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Total Transactions", f"{total:,}")
        k2.metric("Fraudulent", f"{len(fraud):,}", delta=f"{fraud_rate:.1%}")
        k3.metric("Legitimate", f"{len(legit):,}")
        k4.metric("Avg Fraud Amount", f"${fraud['amount'].mean():.2f}")
        k5.metric("Avg Legit Amount", f"${legit['amount'].mean():.2f}")

        st.divider()

        # Charts row 1
        c1, c2, c3 = st.columns(3)
        with c1:
            fig, ax = plt.subplots(figsize=(5,3))
            counts = df["is_fraud"].value_counts()
            ax.pie(counts, labels=["Legitimate","Fraud"], autopct="%1.1f%%",
                   colors=["#4CAF50","#F44336"], startangle=90)
            ax.set_title("Fraud Distribution")
            st.pyplot(fig); plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(5,3))
            fraud_by_cat = df.groupby("merchant_category")["is_fraud"].mean().sort_values()
            fraud_by_cat.plot(kind="barh", ax=ax, color="#F44336")
            ax.set_title("Fraud Rate by Category")
            ax.set_xlabel("Fraud Rate")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c3:
            fig, ax = plt.subplots(figsize=(5,3))
            fraud_by_hour = df.groupby("hour")["is_fraud"].mean()
            ax.plot(fraud_by_hour.index, fraud_by_hour.values, color="#1976D2", linewidth=2)
            ax.fill_between(fraud_by_hour.index, fraud_by_hour.values, alpha=0.2, color="#1976D2")
            ax.set_title("Fraud Rate by Hour")
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Fraud Rate")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Charts row 2
        c4, c5 = st.columns(2)
        with c4:
            fig, ax = plt.subplots(figsize=(6,3))
            legit["amount"].hist(bins=50, alpha=0.6, color="#4CAF50", label="Legit", ax=ax)
            fraud["amount"].hist(bins=50, alpha=0.6, color="#F44336", label="Fraud", ax=ax)
            ax.set_title("Amount Distribution"); ax.set_xlabel("Amount ($)"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c5:
            fig, ax = plt.subplots(figsize=(6,3))
            fraud_by_country = df.groupby("country")["is_fraud"].mean().sort_values(ascending=False)
            fraud_by_country.plot(kind="bar", ax=ax, color="#FF7043")
            ax.set_title("Fraud Rate by Country"); ax.set_xlabel("")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        num_cols = ["amount","hour","distance_from_home_km","ip_risk_score",
                    "num_transactions_last_hour","account_age_days","amount_vs_user_avg","is_fraud"]
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Raw data
        st.subheader("Transaction Data")
        filter_fraud = st.checkbox("Show fraud only")
        display_df = fraud if filter_fraud else df
        st.dataframe(
            display_df.style.apply(
                lambda row: ["background-color:#ffe0e0" if row["is_fraud"]==1 else "" for _ in row], axis=1
            ), use_container_width=True, height=300
        )
        csv = display_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv, "transactions.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – SINGLE PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔎 Analyze a Single Transaction")
    if st.session_state.model is None:
        st.info("Train a model first.")
    else:
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                amount       = st.number_input("Amount ($)", 0.0, 50000.0, 250.0, 10.0)
                hour         = st.slider("Hour of Day", 0, 23, 14)
                day_of_week  = st.selectbox("Day", list(range(7)),
                                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
                merchant_cat = st.selectbox("Merchant Category", MERCHANT_CATEGORIES)
            with c2:
                card_type    = st.selectbox("Card Type", CARD_TYPES)
                device_type  = st.selectbox("Device Type", DEVICE_TYPES)
                country      = st.selectbox("Country", COUNTRIES)
                distance     = st.number_input("Distance from Home (km)", 0.0, 5000.0, 10.0)
            with c3:
                ip_risk      = st.slider("IP Risk Score", 0.0, 1.0, 0.1, 0.01)
                is_new_dev   = st.selectbox("New Device?", [0,1], format_func=lambda x: "Yes" if x else "No")
                acct_age     = st.number_input("Account Age (days)", 1, 3650, 365)
                failed_att   = st.number_input("Failed Attempts (last hr)", 0, 20, 0)
                num_txn_hr   = st.number_input("Txns Last Hour", 0, 50, 1)
                num_txn_day  = st.number_input("Txns Last Day", 0, 200, 5)
                amt_vs_avg   = st.number_input("Amount vs User Avg (ratio)", 0.0, 50.0, 1.0, 0.1)

            submitted = st.form_submit_button("🔍 Analyze", type="primary")

        if submitted:
            txn = {
                "amount": amount, "hour": hour, "day_of_week": day_of_week,
                "merchant_category": merchant_cat, "card_type": card_type,
                "device_type": device_type, "country": country,
                "distance_from_home_km": distance,
                "is_foreign_transaction": int(country != "US"),
                "is_high_risk_country": int(country in ["NG","RU","CN"]),
                "num_transactions_last_hour": num_txn_hr,
                "num_transactions_last_day": num_txn_day,
                "ip_risk_score": ip_risk, "is_new_device": is_new_dev,
                "account_age_days": acct_age,
                "failed_attempts_last_hour": failed_att,
                "amount_vs_user_avg": amt_vs_avg,
            }
            pred, prob, _ = predict_transaction(txn, st.session_state.model,
                                                st.session_state.scaler, st.session_state.encoders)
            flagged = prob >= fraud_threshold

            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Fraud Probability", f"{prob*100:.2f}%")
            r2.metric("Decision", "🚨 FRAUD" if flagged else "✅ LEGITIMATE")
            r3.metric("Threshold Used", f"{fraud_threshold*100:.0f}%")

            if flagged:
                st.error(f"🚨 Transaction flagged as FRAUDULENT (confidence: {prob*100:.1f}%)")
            else:
                st.success(f"✅ Transaction appears LEGITIMATE (fraud prob: {prob*100:.1f}%)")

            # Risk bar
            fig, ax = plt.subplots(figsize=(7,1.2))
            color = "#F44336" if prob > 0.7 else "#FF9800" if prob > fraud_threshold else "#4CAF50"
            ax.barh(["Risk"], [prob], color=color, height=0.5)
            ax.barh(["Risk"], [1-prob], left=[prob], color="#e0e0e0", height=0.5)
            ax.axvline(fraud_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold {fraud_threshold:.0%}")
            ax.set_xlim(0,1); ax.set_title("Fraud Risk Score"); ax.legend(loc="lower right")
            ax.set_xticks([0,0.25,0.5,0.75,1]); ax.set_xticklabels(["0%","25%","50%","75%","100%"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Risk factors
            st.subheader("Risk Factor Breakdown")
            factors = {
                "High Amount": int(amount > 800),
                "Foreign Transaction": int(country != "US"),
                "High Risk Country": int(country in ["NG","RU","CN"]),
                "Late Night (0-3am)": int(hour in [0,1,2,3]),
                "High Velocity": int(num_txn_hr > 5),
                "High IP Risk": int(ip_risk > 0.7),
                "New Device": is_new_dev,
                "New Account (<30d)": int(acct_age < 30),
                "Failed Attempts": int(failed_att > 2),
                "Unusual Amount": int(amt_vs_avg > 5),
                "Prepaid Card": int(card_type == "prepaid"),
                "Risky Merchant": int(merchant_cat in ["atm","luxury","gaming"]),
            }
            active = {k:v for k,v in factors.items() if v}
            if active:
                cols = st.columns(min(4, len(active)))
                for i,(k,_) in enumerate(active.items()):
                    cols[i % len(cols)].warning(f"⚠️ {k}")
            else:
                st.success("No significant risk factors detected.")

            # Log alert
            st.session_state.alert_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "amount": amount, "country": country,
                "merchant": merchant_cat, "prob": f"{prob*100:.1f}%",
                "result": "FRAUD" if flagged else "LEGIT"
            })

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – BATCH UPLOAD
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📁 Batch Transaction Scoring")
    if st.session_state.model is None:
        st.info("Train a model first.")
    else:
        st.markdown("Upload a CSV with transaction data. Required columns match the training schema.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if st.button("📥 Use sample data for batch test"):
            sample = generate_transactions(n=200)
            uploaded_df = sample
            st.session_state["batch_df"] = uploaded_df

        if uploaded:
            st.session_state["batch_df"] = pd.read_csv(uploaded)

        if "batch_df" in st.session_state and st.session_state["batch_df"] is not None:
            batch_df = st.session_state["batch_df"].copy()
            preds, probs = batch_predict(batch_df, st.session_state.model,
                                         st.session_state.scaler, st.session_state.encoders)
            batch_df["fraud_probability"] = (probs * 100).round(2)
            batch_df["prediction"] = ["🚨 FRAUD" if p >= fraud_threshold else "✅ LEGIT" for p in probs]

            b1,b2,b3 = st.columns(3)
            flagged_count = (probs >= fraud_threshold).sum()
            b1.metric("Total Scored", len(batch_df))
            b2.metric("Flagged as Fraud", flagged_count)
            b3.metric("Fraud Rate", f"{flagged_count/len(batch_df):.1%}")

            # Prob distribution
            fig, ax = plt.subplots(figsize=(8,3))
            ax.hist(probs, bins=40, color="#1976D2", edgecolor="white")
            ax.axvline(fraud_threshold, color="red", linestyle="--", label=f"Threshold {fraud_threshold:.0%}")
            ax.set_title("Fraud Probability Distribution"); ax.set_xlabel("Probability"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.dataframe(
                batch_df[["transaction_id","amount","merchant_category","country",
                           "fraud_probability","prediction"]].style.apply(
                    lambda row: ["background-color:#ffe0e0" if "FRAUD" in str(row["prediction"]) else "" for _ in row], axis=1
                ), use_container_width=True, height=350
            )
            result_csv = batch_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results", result_csv, "fraud_results.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – LIVE SIMULATION
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔴 Live Transaction Stream Simulation")
    if st.session_state.model is None:
        st.info("Train a model first.")
    else:
        speed = st.slider("Simulation speed (transactions)", 5, 50, 20)
        if st.button("▶️ Run Simulation", type="primary"):
            sim_df = generate_transactions(n=speed, random_state=np.random.randint(0,9999))
            preds, probs = batch_predict(sim_df, st.session_state.model,
                                          st.session_state.scaler, st.session_state.encoders)

            feed_placeholder = st.empty()
            chart_placeholder = st.empty()
            results = []

            for i in range(len(sim_df)):
                row = sim_df.iloc[i]
                prob = probs[i]
                flagged = prob >= fraud_threshold
                results.append({
                    "txn": i+1, "amount": row["amount"],
                    "merchant": row["merchant_category"],
                    "country": row["country"],
                    "prob": prob, "flagged": flagged
                })

                with feed_placeholder.container():
                    st.markdown(f"**Processing transaction {i+1}/{len(sim_df)}**")
                    latest = results[-min(8, len(results)):]
                    for r in reversed(latest):
                        badge = "🚨 FRAUD" if r["flagged"] else "✅ OK"
                        color = "#ffe0e0" if r["flagged"] else "#e0ffe0"
                        st.markdown(
                            f'<div style="background:{color};padding:6px 12px;border-radius:6px;margin:2px 0">'
                            f'Txn #{r["txn"]} | ${r["amount"]:.2f} | {r["merchant"]} | {r["country"]} | '
                            f'Prob: {r["prob"]*100:.1f}% | {badge}</div>',
                            unsafe_allow_html=True
                        )

                with chart_placeholder.container():
                    rdf = pd.DataFrame(results)
                    fig, ax = plt.subplots(figsize=(10,2.5))
                    colors = ["#F44336" if f else "#4CAF50" for f in rdf["flagged"]]
                    ax.bar(rdf["txn"], rdf["prob"], color=colors, width=0.8)
                    ax.axhline(fraud_threshold, color="black", linestyle="--", linewidth=1)
                    ax.set_ylim(0,1); ax.set_xlabel("Transaction #"); ax.set_ylabel("Fraud Prob")
                    ax.set_title("Live Fraud Probability Feed")
                    plt.tight_layout(); chart_placeholder.pyplot(fig); plt.close()

                time.sleep(0.08)

            fraud_sim = sum(1 for r in results if r["flagged"])
            st.success(f"Simulation complete: {fraud_sim}/{len(results)} transactions flagged as fraud.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 – MODEL ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📈 Model Performance & Analytics")
    if st.session_state.metrics is None:
        st.info("Train a model first.")
    else:
        m = st.session_state.metrics
        report = m["report"]

        # Metrics
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Accuracy",        f"{report['accuracy']:.2%}")
        k2.metric("Precision(Fraud)", f"{report['1']['precision']:.2%}")
        k3.metric("Recall(Fraud)",    f"{report['1']['recall']:.2%}")
        k4.metric("F1(Fraud)",        f"{report['1']['f1-score']:.2%}")
        k5.metric("ROC-AUC",          f"{m['roc_auc']:.4f}")

        # CV scores
        cv = m["cv_scores"]
        st.info(f"5-Fold CV ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots()
            sns.heatmap(m["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
            ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            st.markdown("**ROC Curve**")
            fpr, tpr = m["roc_curve"]
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color="#1976D2", lw=2, label=f"AUC={m['roc_auc']:.4f}")
            ax.plot([0,1],[0,1],"k--", alpha=0.4)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Precision-Recall Curve**")
            prec, rec = m["pr_curve"]
            fig, ax = plt.subplots()
            ax.plot(rec, prec, color="#E91E63", lw=2, label=f"AP={m['avg_precision']:.4f}")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curve"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c4:
            st.markdown("**Fraud Probability Distribution**")
            y_test = m["y_test"]; y_prob = m["y_prob"]
            fig, ax = plt.subplots()
            ax.hist(y_prob[y_test==0], bins=40, alpha=0.6, color="#4CAF50", label="Legit")
            ax.hist(y_prob[y_test==1], bins=40, alpha=0.6, color="#F44336", label="Fraud")
            ax.axvline(fraud_threshold, color="black", linestyle="--", label="Threshold")
            ax.set_xlabel("Predicted Probability"); ax.legend()
            ax.set_title("Score Distribution"); plt.tight_layout(); st.pyplot(fig); plt.close()

        # Feature importances
        if m["feature_importances"]:
            st.markdown("**Feature Importances**")
            fi = pd.Series(m["feature_importances"]).sort_values()
            fig, ax = plt.subplots(figsize=(9,5))
            colors = ["#F44336" if v > fi.median() else "#1976D2" for v in fi.values]
            fi.plot(kind="barh", ax=ax, color=colors)
            ax.set_title("Feature Importances (red = above median)")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Model comparison
        st.markdown("**Model Comparison**")
        comp = m["model_comparison"]
        comp_df = pd.DataFrame(comp).T.reset_index().rename(columns={"index":"Model"})
        fig, ax = plt.subplots(figsize=(8,3))
        x = np.arange(len(comp_df))
        ax.bar(x-0.2, comp_df["roc_auc"], 0.35, label="ROC-AUC", color="#1976D2")
        ax.bar(x+0.2, comp_df["avg_precision"], 0.35, label="Avg Precision", color="#E91E63")
        ax.set_xticks(x); ax.set_xticklabels(comp_df["Model"], rotation=15)
        ax.set_ylim(0,1); ax.legend(); ax.set_title("Model Comparison")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Anomaly scores
        st.markdown("**Anomaly Detection (Isolation Forest)**")
        anom = m["anomaly_scores"]; y_t = m["y_test"]
        fig, ax = plt.subplots(figsize=(8,3))
        ax.hist(anom[y_t==0], bins=40, alpha=0.6, color="#4CAF50", label="Legit")
        ax.hist(anom[y_t==1], bins=40, alpha=0.6, color="#F44336", label="Fraud")
        ax.set_xlabel("Anomaly Score (higher = more anomalous)")
        ax.set_title("Isolation Forest Anomaly Scores"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 – ALERT LOG
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🚨 Transaction Alert Log")
    log = st.session_state.alert_log
    if not log:
        st.info("No transactions analyzed yet. Use the Predict tab to analyze transactions.")
    else:
        log_df = pd.DataFrame(log)
        fraud_alerts = log_df[log_df["result"] == "FRAUD"]
        a1,a2,a3 = st.columns(3)
        a1.metric("Total Analyzed", len(log_df))
        a2.metric("Fraud Alerts", len(fraud_alerts))
        a3.metric("Alert Rate", f"{len(fraud_alerts)/len(log_df):.1%}")

        st.dataframe(
            log_df.style.apply(
                lambda row: ["background-color:#ffe0e0" if row["result"]=="FRAUD" else "" for _ in row], axis=1
            ), use_container_width=True
        )
        if st.button("🗑️ Clear Log"):
            st.session_state.alert_log = []
            st.rerun()
        log_csv = log_df.to_csv(index=False).encode()
        st.download_button("⬇️ Export Log", log_csv, "alert_log.csv", "text/csv")
