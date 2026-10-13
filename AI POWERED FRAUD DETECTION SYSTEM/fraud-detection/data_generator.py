import pandas as pd
import numpy as np

MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "restaurant", "online", "atm", "luxury", "gaming"]
CARD_TYPES = ["credit", "debit", "prepaid"]
DEVICE_TYPES = ["mobile", "desktop", "tablet", "pos_terminal"]
COUNTRIES = ["US", "UK", "IN", "NG", "RU", "CN", "BR", "DE", "FR", "AU"]

def generate_transactions(n=1000, random_state=42):
    np.random.seed(random_state)

    n_users = max(50, n // 20)
    user_ids = np.random.randint(1000, 1000 + n_users, size=n)
    avg_spend = {uid: np.random.exponential(150) for uid in range(1000, 1000 + n_users)}

    amounts = np.array([
        np.random.exponential(scale=avg_spend.get(uid, 150))
        for uid in user_ids
    ]).round(2)

    hour = np.random.randint(0, 24, size=n)
    day_of_week = np.random.randint(0, 7, size=n)
    merchant_category = np.random.choice(MERCHANT_CATEGORIES, size=n)
    card_type = np.random.choice(CARD_TYPES, size=n, p=[0.5, 0.35, 0.15])
    device_type = np.random.choice(DEVICE_TYPES, size=n, p=[0.45, 0.3, 0.1, 0.15])
    country = np.random.choice(COUNTRIES, size=n, p=[0.4, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.08, 0.07, 0.05])
    is_foreign = (country != "US").astype(int)
    distance_from_home = np.random.exponential(scale=30, size=n).round(2)
    num_txn_last_hour = np.random.poisson(lam=2, size=n)
    num_txn_last_day = num_txn_last_hour * np.random.randint(3, 10, size=n)
    ip_risk_score = np.random.beta(1, 5, size=n).round(4)           # 0-1, higher = riskier
    is_new_device = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    is_high_risk_country = np.isin(country, ["NG", "RU", "CN"]).astype(int)
    account_age_days = np.random.randint(1, 3650, size=n)
    failed_attempts_last_hour = np.random.poisson(lam=0.3, size=n)
    amount_vs_avg = np.array([
        amounts[i] / max(avg_spend.get(user_ids[i], 150), 1)
        for i in range(n)
    ]).round(4)

    df = pd.DataFrame({
        "transaction_id": range(1, n + 1),
        "user_id": user_ids,
        "amount": amounts,
        "hour": hour,
        "day_of_week": day_of_week,
        "merchant_category": merchant_category,
        "card_type": card_type,
        "device_type": device_type,
        "country": country,
        "distance_from_home_km": distance_from_home,
        "is_foreign_transaction": is_foreign,
        "is_high_risk_country": is_high_risk_country,
        "num_transactions_last_hour": num_txn_last_hour,
        "num_transactions_last_day": num_txn_last_day,
        "ip_risk_score": ip_risk_score,
        "is_new_device": is_new_device,
        "account_age_days": account_age_days,
        "failed_attempts_last_hour": failed_attempts_last_hour,
        "amount_vs_user_avg": amount_vs_avg,
    })

    # Realistic fraud scoring
    fraud_score = (
        (df["amount"] > 800).astype(int) * 2 +
        df["is_foreign_transaction"] * 1.5 +
        df["is_high_risk_country"] * 2 +
        (df["hour"].isin([0, 1, 2, 3])).astype(int) +
        (df["num_transactions_last_hour"] > 5).astype(int) * 2 +
        (df["distance_from_home_km"] > 100).astype(int) +
        (df["ip_risk_score"] > 0.7).astype(int) * 2 +
        df["is_new_device"] * 1.5 +
        (df["account_age_days"] < 30).astype(int) * 2 +
        (df["failed_attempts_last_hour"] > 2).astype(int) * 2 +
        (df["amount_vs_user_avg"] > 5).astype(int) * 2 +
        (df["card_type"] == "prepaid").astype(int) +
        (df["merchant_category"].isin(["atm", "luxury", "gaming"])).astype(int)
    )

    df["is_fraud"] = (
        fraud_score + np.random.binomial(1, 0.04, size=n) >= 6
    ).astype(int)

    return df
