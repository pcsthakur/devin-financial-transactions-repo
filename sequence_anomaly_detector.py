"""
Sequence Anomaly Detection for Financial Transactions.

Groups transactions by customer, sorts by time step, and analyzes
transaction sequences to detect suspicious patterns such as:
  - Repeated high-value transactions
  - TRANSFER followed by CASH_OUT
  - Sudden increases in transaction amounts

Outputs a sequence anomaly report with customer_id, sequence_pattern,
anomaly_level, and explanation.
"""

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_AMOUNT_THRESHOLD = 10_000
REPEATED_HIGH_VALUE_MIN_COUNT = 2

# Amount increase factor that counts as "sudden"
SUDDEN_INCREASE_FACTOR = 5.0

# Anomaly score thresholds (aligned with Fraud Risk Scoring Guidelines)
LOW_THRESHOLD = 40
HIGH_THRESHOLD = 70

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Example1.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "sequence_anomaly_report.csv")


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------


def load_transactions(path: str) -> pd.DataFrame:
    """Load the transaction CSV into a DataFrame."""
    df = pd.read_csv(path)
    return df


def group_and_sort(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group transactions by originating customer and sort by step (time).

    Returns a dict mapping customer_id -> sorted DataFrame of their
    transactions.
    """
    grouped: dict[str, pd.DataFrame] = {}
    for customer_id, group_df in df.groupby("nameOrig"):
        sorted_df = group_df.sort_values("step").reset_index(drop=True)
        grouped[str(customer_id)] = sorted_df
    return grouped


# ---------------------------------------------------------------------------
# Pattern detection helpers
# ---------------------------------------------------------------------------


def detect_repeated_high_value(txns: pd.DataFrame) -> tuple[int, list[str]]:
    """Detect repeated high-value transactions (amount > threshold).

    Returns (score_contribution, list_of_reasons).
    """
    high_value_txns = txns[txns["amount"] > HIGH_AMOUNT_THRESHOLD]
    count = len(high_value_txns)
    if count >= REPEATED_HIGH_VALUE_MIN_COUNT:
        total_amount = high_value_txns["amount"].sum()
        score = min(15 + (count - REPEATED_HIGH_VALUE_MIN_COUNT) * 10, 40)
        reason = (
            f"Repeated high-value transactions: {count} transactions "
            f"above ${HIGH_AMOUNT_THRESHOLD:,} totaling ${total_amount:,.2f}"
        )
        return score, [reason]
    return 0, []


def detect_transfer_then_cashout(txns: pd.DataFrame) -> tuple[int, list[str]]:
    """Detect TRANSFER followed by CASH_OUT pattern in the sequence.

    This is a classic layering pattern where money is moved via transfer
    and then immediately cashed out.

    Returns (score_contribution, list_of_reasons).
    """
    types = txns["type"].tolist()
    amounts = txns["amount"].tolist()
    steps = txns["step"].tolist()
    score = 0
    reasons: list[str] = []

    for i in range(len(types) - 1):
        if types[i] == "TRANSFER" and types[i + 1] == "CASH_OUT":
            pair_score = 30
            # Boost if the amounts are similar (layering indicator)
            if amounts[i] > 0 and abs(amounts[i] - amounts[i + 1]) / amounts[i] < 0.2:
                pair_score += 10
                reasons.append(
                    f"TRANSFER (${amounts[i]:,.2f}) at step {steps[i]} followed by "
                    f"CASH_OUT (${amounts[i + 1]:,.2f}) at step {steps[i + 1]} "
                    f"with similar amounts (layering indicator)"
                )
            else:
                reasons.append(
                    f"TRANSFER (${amounts[i]:,.2f}) at step {steps[i]} followed by "
                    f"CASH_OUT (${amounts[i + 1]:,.2f}) at step {steps[i + 1]}"
                )
            score = max(score, pair_score)

    return score, reasons


def detect_sudden_amount_increase(txns: pd.DataFrame) -> tuple[int, list[str]]:
    """Detect sudden increases in transaction amounts within a customer's sequence.

    A sudden increase is when a transaction amount is >= SUDDEN_INCREASE_FACTOR
    times the previous transaction amount, and the new amount exceeds the
    high-amount threshold.

    Returns (score_contribution, list_of_reasons).
    """
    amounts = txns["amount"].tolist()
    types = txns["type"].tolist()
    steps = txns["step"].tolist()
    score = 0
    reasons: list[str] = []

    for i in range(1, len(amounts)):
        prev_amount = amounts[i - 1]
        curr_amount = amounts[i]
        if (
            prev_amount > 0
            and curr_amount >= prev_amount * SUDDEN_INCREASE_FACTOR
            and curr_amount > HIGH_AMOUNT_THRESHOLD
        ):
            increase_factor = curr_amount / prev_amount
            spike_score = min(int(10 + increase_factor * 2), 35)
            reasons.append(
                f"Sudden amount increase: ${prev_amount:,.2f} ({types[i - 1]}, step {steps[i - 1]}) "
                f"-> ${curr_amount:,.2f} ({types[i]}, step {steps[i]}) "
                f"({increase_factor:.1f}x increase)"
            )
            score = max(score, spike_score)

    return score, reasons


def detect_rapid_sequence(txns: pd.DataFrame) -> tuple[int, list[str]]:
    """Detect rapid sequences of transactions in the same time step.

    Multiple transactions from the same customer in a single step
    indicate automated or suspicious behavior.

    Returns (score_contribution, list_of_reasons).
    """
    step_counts = txns["step"].value_counts()
    rapid_steps = step_counts[step_counts > 1]

    if rapid_steps.empty:
        return 0, []

    max_count = int(rapid_steps.max())
    score = min(10 + (max_count - 2) * 5, 25)
    reasons = []
    for step_val, count in rapid_steps.items():
        step_txns = txns[txns["step"] == step_val]
        total = step_txns["amount"].sum()
        reasons.append(
            f"Rapid burst: {int(count)} transactions in step {step_val} "
            f"totaling ${total:,.2f}"
        )
    return score, reasons


def detect_balance_drain_sequence(txns: pd.DataFrame) -> tuple[int, list[str]]:
    """Detect sequences where the account balance is progressively drained.

    Returns (score_contribution, list_of_reasons).
    """
    if len(txns) < 2:
        return 0, []

    first_balance = txns.iloc[0]["oldbalanceOrg"]
    last_balance = txns.iloc[-1]["newbalanceOrig"]

    if first_balance > 0 and last_balance == 0:
        total_moved = txns["amount"].sum()
        score = 20
        reason = (
            f"Progressive balance drain: balance went from "
            f"${first_balance:,.2f} to $0.00 across {len(txns)} transactions "
            f"(total moved: ${total_moved:,.2f})"
        )
        return score, [reason]

    return 0, []


# ---------------------------------------------------------------------------
# Sequence pattern summarizer
# ---------------------------------------------------------------------------


def build_sequence_pattern(txns: pd.DataFrame) -> str:
    """Build a compact string representing the transaction type sequence.

    Example: "PAYMENT -> TRANSFER -> CASH_OUT" or "PAYMENT x3 -> TRANSFER -> CASH_OUT"
    """
    types = txns["type"].tolist()
    if not types:
        return ""

    compressed: list[str] = []
    current_type = types[0]
    count = 1

    for t in types[1:]:
        if t == current_type:
            count += 1
        else:
            if count > 1:
                compressed.append(f"{current_type} x{count}")
            else:
                compressed.append(current_type)
            current_type = t
            count = 1

    if count > 1:
        compressed.append(f"{current_type} x{count}")
    else:
        compressed.append(current_type)

    return " -> ".join(compressed)


# ---------------------------------------------------------------------------
# Anomaly level assignment
# ---------------------------------------------------------------------------


def assign_anomaly_level(score: int) -> str:
    """Assign anomaly level based on composite score.

    Uses the risk guidelines:
      - LOW:    score < 40
      - MEDIUM: 40 <= score <= 70
      - HIGH:   score > 70
    """
    if score > HIGH_THRESHOLD:
        return "HIGH"
    if score >= LOW_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def analyze_customer_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze transaction sequences for each customer and detect anomalies.

    Returns a DataFrame with columns:
      - customer_id
      - transaction_count
      - sequence_pattern
      - anomaly_score
      - anomaly_level
      - explanation
    """
    grouped = group_and_sort(df)
    records: list[dict[str, object]] = []

    for customer_id, txns in grouped.items():
        total_score = 0
        all_reasons: list[str] = []

        # 1. Repeated high-value transactions
        pts, reasons = detect_repeated_high_value(txns)
        total_score += pts
        all_reasons.extend(reasons)

        # 2. TRANSFER followed by CASH_OUT
        pts, reasons = detect_transfer_then_cashout(txns)
        total_score += pts
        all_reasons.extend(reasons)

        # 3. Sudden increase in transaction amounts
        pts, reasons = detect_sudden_amount_increase(txns)
        total_score += pts
        all_reasons.extend(reasons)

        # 4. Rapid transaction bursts
        pts, reasons = detect_rapid_sequence(txns)
        total_score += pts
        all_reasons.extend(reasons)

        # 5. Progressive balance drain
        pts, reasons = detect_balance_drain_sequence(txns)
        total_score += pts
        all_reasons.extend(reasons)

        # Cap score at 100
        total_score = min(total_score, 100)

        sequence_pattern = build_sequence_pattern(txns)
        anomaly_level = assign_anomaly_level(total_score)
        explanation = (
            "; ".join(all_reasons) if all_reasons else "No anomalous patterns detected"
        )

        records.append(
            {
                "customer_id": customer_id,
                "transaction_count": len(txns),
                "sequence_pattern": sequence_pattern,
                "anomaly_score": total_score,
                "anomaly_level": anomaly_level,
                "explanation": explanation,
            }
        )

    report_df = pd.DataFrame(records)
    report_df = report_df.sort_values("anomaly_score", ascending=False).reset_index(
        drop=True
    )
    return report_df


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(report_df: pd.DataFrame, output_path: str) -> None:
    """Write the sequence anomaly report to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_df.to_csv(output_path, index=False)


def print_summary(report_df: pd.DataFrame) -> None:
    """Print a human-readable summary of the anomaly analysis."""
    total = len(report_df)
    high = len(report_df[report_df["anomaly_level"] == "HIGH"])
    medium = len(report_df[report_df["anomaly_level"] == "MEDIUM"])
    low = len(report_df[report_df["anomaly_level"] == "LOW"])

    print("\n" + "=" * 70)
    print("SEQUENCE ANOMALY DETECTION SUMMARY")
    print("=" * 70)
    print(f"Total customers analyzed: {total}")
    print(f"  HIGH anomaly:   {high:>3} ({high / total * 100:.1f}%)")
    print(f"  MEDIUM anomaly: {medium:>3} ({medium / total * 100:.1f}%)")
    print(f"  LOW anomaly:    {low:>3} ({low / total * 100:.1f}%)")
    print("=" * 70)

    if high > 0:
        print("\nTop HIGH anomaly customers:")
        top = report_df[report_df["anomaly_level"] == "HIGH"].head(10)
        for _, row in top.iterrows():
            print(
                f"  {row['customer_id']} | "
                f"Score: {row['anomaly_score']:>3} | "
                f"Txns: {row['transaction_count']:>2} | "
                f"Pattern: {row['sequence_pattern']}"
            )
            print(f"    Explanation: {row['explanation']}")

    if medium > 0:
        print("\nTop MEDIUM anomaly customers:")
        top = report_df[report_df["anomaly_level"] == "MEDIUM"].head(5)
        for _, row in top.iterrows():
            print(
                f"  {row['customer_id']} | "
                f"Score: {row['anomaly_score']:>3} | "
                f"Txns: {row['transaction_count']:>2} | "
                f"Pattern: {row['sequence_pattern']}"
            )
            print(f"    Explanation: {row['explanation']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load data, analyze sequences, and generate anomaly report."""
    print(f"Loading transaction data from {DATA_PATH} ...")
    df = load_transactions(DATA_PATH)
    print(f"Loaded {len(df)} transactions.\n")

    print("Grouping by customer and analyzing transaction sequences ...")
    report_df = analyze_customer_sequences(df)

    print(f"\nGenerating report at {REPORT_PATH} ...")
    generate_report(report_df, REPORT_PATH)

    print_summary(report_df)
    print(f"\nFull report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
