"""
Fraud Risk Scoring System for Financial Transactions.

Analyzes each transaction independently and assigns a risk score (0-100)
based on multiple risk signals including transaction amount, type,
destination account patterns, and balance anomalies.
"""

import os

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_AMOUNT_THRESHOLD = 10_000
VERY_HIGH_AMOUNT_THRESHOLD = 100_000
EXTREME_AMOUNT_THRESHOLD = 500_000

RISKY_TYPES = {"CASH_OUT": 20, "TRANSFER": 15}

RISK_LEVEL_THRESHOLDS = {"LOW": 40, "MEDIUM": 70}

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Example1.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "fraud_risk_report.csv")


# ---------------------------------------------------------------------------
# Risk signal functions
# ---------------------------------------------------------------------------


def _score_transaction_amount(amount: float) -> tuple[int, str]:
    """Score based on transaction amount (0-30 points)."""
    if amount > EXTREME_AMOUNT_THRESHOLD:
        return 30, f"Extreme transaction amount ({amount:,.2f} > {EXTREME_AMOUNT_THRESHOLD:,})"
    if amount > VERY_HIGH_AMOUNT_THRESHOLD:
        return 25, f"Very high transaction amount ({amount:,.2f} > {VERY_HIGH_AMOUNT_THRESHOLD:,})"
    if amount > HIGH_AMOUNT_THRESHOLD:
        return 15, f"High transaction amount ({amount:,.2f} > {HIGH_AMOUNT_THRESHOLD:,})"
    return 0, ""


def _score_transaction_type(txn_type: str) -> tuple[int, str]:
    """Score based on transaction type (0-20 points)."""
    points = RISKY_TYPES.get(txn_type, 0)
    if points > 0:
        return points, f"Risky transaction type ({txn_type})"
    return 0, ""


def _score_destination_frequency(dest: str, dest_counts: dict[str, int]) -> tuple[int, str]:
    """Score based on how many times a destination account receives funds (0-15 points)."""
    count = dest_counts.get(dest, 1)
    if count > 1:
        points = min(5 + (count - 1) * 3, 15)
        return points, f"Repeated destination account {dest} ({count} incoming transactions)"
    return 0, ""


def _score_balance_anomalies(
    amount: float,
    old_balance_org: float,
    new_balance_org: float,
    old_balance_dest: float,
) -> tuple[int, str]:
    """Score based on balance anomalies (0-25 points)."""
    points = 0
    reasons = []

    # Origin has zero balance but sends money
    if old_balance_org == 0 and amount > 0:
        points += 10
        reasons.append("Origin account has zero balance but initiated a transaction")

    # Entire balance drained
    if old_balance_org > 0 and new_balance_org == 0:
        points += 10
        reasons.append("Entire origin balance drained to zero")

    # Destination has zero old balance (new/unseen account)
    if old_balance_dest == 0:
        points += 5
        reasons.append("Destination account had zero prior balance (new/unseen account)")

    return min(points, 25), "; ".join(reasons)


def _score_amount_vs_balance(amount: float, old_balance_org: float) -> tuple[int, str]:
    """Score when transaction amount exceeds the origin balance (0-10 points)."""
    if old_balance_org > 0 and amount > old_balance_org:
        return 10, f"Transaction amount ({amount:,.2f}) exceeds origin balance ({old_balance_org:,.2f})"
    return 0, ""


def _score_rapid_transactions(
    name_orig: str, step: int, orig_step_counts: dict[tuple[str, int], int]
) -> tuple[int, str]:
    """Score for multiple transactions from the same account in the same step (0-10 points)."""
    count = orig_step_counts.get((name_orig, step), 1)
    if count > 1:
        return 10, f"Multiple transactions ({count}) from account {name_orig} in the same time step"
    return 0, ""


# ---------------------------------------------------------------------------
# Main scoring logic
# ---------------------------------------------------------------------------


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fraud risk scores for every transaction in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data with columns: step, type, amount, nameOrig,
        oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest,
        newbalanceDest, isFraud, isFlaggedFraud.

    Returns
    -------
    pd.DataFrame
        Report with columns: transaction_id, risk_score, risk_level, explanation.
    """
    # Pre-compute aggregate lookups
    dest_counts = df["nameDest"].value_counts().to_dict()
    orig_step_counts = df.groupby(["nameOrig", "step"]).size().to_dict()

    records: list[dict[str, object]] = []

    for idx, row in df.iterrows():
        reasons: list[str] = []
        total_score = 0

        # 1. Transaction amount
        pts, reason = _score_transaction_amount(row["amount"])
        total_score += pts
        if reason:
            reasons.append(reason)

        # 2. Transaction type
        pts, reason = _score_transaction_type(row["type"])
        total_score += pts
        if reason:
            reasons.append(reason)

        # 3. Destination account frequency
        pts, reason = _score_destination_frequency(row["nameDest"], dest_counts)
        total_score += pts
        if reason:
            reasons.append(reason)

        # 4. Balance anomalies
        pts, reason = _score_balance_anomalies(
            row["amount"],
            row["oldbalanceOrg"],
            row["newbalanceOrig"],
            row["oldbalanceDest"],
        )
        total_score += pts
        if reason:
            reasons.append(reason)

        # 5. Amount vs balance
        pts, reason = _score_amount_vs_balance(row["amount"], row["oldbalanceOrg"])
        total_score += pts
        if reason:
            reasons.append(reason)

        # 6. Rapid transactions
        pts, reason = _score_rapid_transactions(
            row["nameOrig"], row["step"], orig_step_counts
        )
        total_score += pts
        if reason:
            reasons.append(reason)

        # Cap at 100
        total_score = min(total_score, 100)

        # Assign risk level
        if total_score > RISK_LEVEL_THRESHOLDS["MEDIUM"]:
            risk_level = "HIGH"
        elif total_score >= RISK_LEVEL_THRESHOLDS["LOW"]:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        explanation = "; ".join(reasons) if reasons else "No significant risk signals detected"

        records.append(
            {
                "transaction_id": f"TXN-{idx + 1:04d}",
                "risk_score": total_score,
                "risk_level": risk_level,
                "explanation": explanation,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(report_df: pd.DataFrame, output_path: str) -> None:
    """Write the risk report DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_df.to_csv(output_path, index=False)


def main() -> None:
    """Entry point: load data, score transactions, and write report."""
    print("Loading transaction data ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} transactions.\n")

    print("Computing risk scores ...")
    report_df = compute_risk_scores(df)

    print("\n--- Risk Distribution ---")
    print(report_df["risk_level"].value_counts().to_string())

    print(f"\nGenerating report at {REPORT_PATH} ...")
    generate_report(report_df, REPORT_PATH)

    print("\nSample output (first 10 rows):")
    print(report_df.head(10).to_string(index=False))
    print(f"\nDone. Full report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
