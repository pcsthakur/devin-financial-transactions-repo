"""
Fraud Risk Scoring System

Analyzes financial transactions and assigns risk scores (0-100) based on
multiple risk signals including transaction amount, type, account behavior,
and balance patterns. Generates a transaction-level risk report.
"""

import os

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join("data", "Example1.csv")
OUTPUT_DIR = "reports"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fraud_risk_report.csv")

HIGH_AMOUNT_THRESHOLD = 10000
RISKY_TYPES = {"CASH_OUT", "TRANSFER"}

# Risk level thresholds
LOW_THRESHOLD = 40
HIGH_THRESHOLD = 70

# Scoring weights (sum to ~100 at maximum)
WEIGHT_AMOUNT = 25
WEIGHT_TYPE = 20
WEIGHT_BALANCE_DRAIN = 15
WEIGHT_DEST_FREQUENCY = 15
WEIGHT_ZERO_ORIG_BALANCE = 10
WEIGHT_DEST_BALANCE_DROP = 10
WEIGHT_AMOUNT_VS_BALANCE = 5


# ---------------------------------------------------------------------------
# Risk Signal Functions
# ---------------------------------------------------------------------------


def score_amount(amount: float) -> float:
    """Score based on transaction amount. Higher amounts are riskier."""
    if amount > 500000:
        return WEIGHT_AMOUNT
    if amount > 100000:
        return WEIGHT_AMOUNT * 0.8
    if amount > HIGH_AMOUNT_THRESHOLD:
        return WEIGHT_AMOUNT * 0.5
    if amount > 5000:
        return WEIGHT_AMOUNT * 0.2
    return 0


def score_type(txn_type: str) -> float:
    """Score based on transaction type. CASH_OUT and TRANSFER are riskier."""
    if txn_type in RISKY_TYPES:
        return WEIGHT_TYPE
    if txn_type == "DEBIT":
        return WEIGHT_TYPE * 0.3
    return 0


def score_balance_drain(old_balance: float, new_balance: float) -> float:
    """Score when an account is fully drained (old > 0, new == 0)."""
    if old_balance > 0 and new_balance == 0:
        return WEIGHT_BALANCE_DRAIN
    if old_balance > 0 and new_balance < old_balance * 0.1:
        return WEIGHT_BALANCE_DRAIN * 0.5
    return 0


def score_dest_frequency(dest_account: str, dest_counts: dict) -> float:
    """Score based on how often a destination account receives funds.
    Accounts receiving many transactions may be collection points."""
    count = dest_counts.get(dest_account, 1)
    if count >= 5:
        return WEIGHT_DEST_FREQUENCY
    if count >= 3:
        return WEIGHT_DEST_FREQUENCY * 0.6
    if count >= 2:
        return WEIGHT_DEST_FREQUENCY * 0.3
    return 0


def score_zero_orig_balance(old_balance: float) -> float:
    """Score when originating account has zero balance (possible synthetic
    or compromised account)."""
    if old_balance == 0:
        return WEIGHT_ZERO_ORIG_BALANCE
    return 0


def score_dest_balance_drop(old_dest_balance: float, new_dest_balance: float) -> float:
    """Score when destination account balance drops despite receiving funds
    (rapid movement of money)."""
    if old_dest_balance > 0 and new_dest_balance == 0:
        return WEIGHT_DEST_BALANCE_DROP
    return 0


def score_amount_vs_balance(amount: float, old_balance: float) -> float:
    """Score when transaction amount exceeds or equals the full account balance
    (complete withdrawal pattern)."""
    if old_balance > 0 and amount >= old_balance:
        return WEIGHT_AMOUNT_VS_BALANCE
    return 0


# ---------------------------------------------------------------------------
# Risk Level Assignment
# ---------------------------------------------------------------------------


def assign_risk_level(score: float) -> str:
    """Assign a risk category based on the numeric score."""
    if score > HIGH_THRESHOLD:
        return "HIGH"
    if score >= LOW_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Explanation Builder
# ---------------------------------------------------------------------------


def build_explanation(row: pd.Series, dest_counts: dict) -> str:
    """Build a human-readable explanation of the risk signals detected."""
    reasons = []

    if row["amount"] > HIGH_AMOUNT_THRESHOLD:
        reasons.append(f"High transaction amount (${row['amount']:,.2f})")

    if row["type"] in RISKY_TYPES:
        reasons.append(f"Risky transaction type ({row['type']})")

    if row["oldbalanceOrg"] > 0 and row["newbalanceOrig"] == 0:
        reasons.append("Account fully drained after transaction")
    elif row["oldbalanceOrg"] > 0 and row["newbalanceOrig"] < row["oldbalanceOrg"] * 0.1:
        reasons.append("Account nearly drained (>90% withdrawn)")

    dest_count = dest_counts.get(row["nameDest"], 1)
    if dest_count >= 2:
        reasons.append(
            f"Destination account ({row['nameDest']}) received {dest_count} transactions"
        )

    if row["oldbalanceOrg"] == 0:
        reasons.append("Originating account had zero balance")

    if row["oldbalanceDest"] > 0 and row["newbalanceDest"] == 0:
        reasons.append("Destination account balance dropped to zero despite receiving funds")

    if row["oldbalanceOrg"] > 0 and row["amount"] >= row["oldbalanceOrg"]:
        reasons.append("Transaction amount equals or exceeds full account balance")

    if not reasons:
        reasons.append("No significant risk signals detected")

    return "; ".join(reasons)


# ---------------------------------------------------------------------------
# Main Scoring Pipeline
# ---------------------------------------------------------------------------


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute risk scores for each transaction in the DataFrame."""
    # Pre-compute destination account frequency
    dest_counts = df["nameDest"].value_counts().to_dict()

    results = []
    for idx, row in df.iterrows():
        risk_score = (
            score_amount(row["amount"])
            + score_type(row["type"])
            + score_balance_drain(row["oldbalanceOrg"], row["newbalanceOrig"])
            + score_dest_frequency(row["nameDest"], dest_counts)
            + score_zero_orig_balance(row["oldbalanceOrg"])
            + score_dest_balance_drop(row["oldbalanceDest"], row["newbalanceDest"])
            + score_amount_vs_balance(row["amount"], row["oldbalanceOrg"])
        )

        # Cap score at 100
        risk_score = min(risk_score, 100)

        risk_level = assign_risk_level(risk_score)
        explanation = build_explanation(row, dest_counts)

        results.append(
            {
                "transaction_id": idx + 1,
                "type": row["type"],
                "amount": row["amount"],
                "nameOrig": row["nameOrig"],
                "nameDest": row["nameDest"],
                "risk_score": round(risk_score, 2),
                "risk_level": risk_level,
                "explanation": explanation,
            }
        )

    return pd.DataFrame(results)


def generate_report(df_scores: pd.DataFrame, output_path: str) -> None:
    """Save the scored transactions to a CSV report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scores.to_csv(output_path, index=False)
    print(f"Report saved to {output_path}")


def print_summary(df_scores: pd.DataFrame) -> None:
    """Print a summary of the risk analysis."""
    total = len(df_scores)
    high = len(df_scores[df_scores["risk_level"] == "HIGH"])
    medium = len(df_scores[df_scores["risk_level"] == "MEDIUM"])
    low = len(df_scores[df_scores["risk_level"] == "LOW"])

    print("\n" + "=" * 60)
    print("FRAUD RISK SCORING SUMMARY")
    print("=" * 60)
    print(f"Total transactions analyzed: {total}")
    print(f"  HIGH risk:   {high:>3} ({high / total * 100:.1f}%)")
    print(f"  MEDIUM risk: {medium:>3} ({medium / total * 100:.1f}%)")
    print(f"  LOW risk:    {low:>3} ({low / total * 100:.1f}%)")
    print("=" * 60)

    if high > 0:
        print("\nTop HIGH risk transactions:")
        top = df_scores[df_scores["risk_level"] == "HIGH"].sort_values(
            "risk_score", ascending=False
        )
        for _, row in top.head(10).iterrows():
            print(
                f"  TX #{row['transaction_id']:>3} | "
                f"Score: {row['risk_score']:>5} | "
                f"Type: {row['type']:<8} | "
                f"Amount: ${row['amount']:>14,.2f} | "
                f"{row['explanation']}"
            )


def main() -> None:
    """Load data, compute risk scores, and generate report."""
    print(f"Loading transactions from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} transactions.")

    print("Computing fraud risk scores...")
    df_scores = compute_risk_scores(df)

    generate_report(df_scores, OUTPUT_PATH)
    print_summary(df_scores)


if __name__ == "__main__":
    main()
