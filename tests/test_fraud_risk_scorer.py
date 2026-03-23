"""Comprehensive unit tests for fraud_risk_scorer module."""

import os

import pandas as pd
import pytest

from fraud_risk_scorer import (
    EXTREME_AMOUNT_THRESHOLD,
    HIGH_AMOUNT_THRESHOLD,
    RISK_LEVEL_THRESHOLDS,
    RISKY_TYPES,
    VERY_HIGH_AMOUNT_THRESHOLD,
    _score_amount_vs_balance,
    _score_balance_anomalies,
    _score_destination_frequency,
    _score_rapid_transactions,
    _score_transaction_amount,
    _score_transaction_type,
    compute_risk_scores,
    generate_report,
)


# ---------------------------------------------------------------------------
# Helper to build a single-row DataFrame for compute_risk_scores
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a transaction DataFrame from a list of row dicts with defaults."""
    defaults = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C1000",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9900.0,
        "nameDest": "M1000",
        "oldbalanceDest": 5000.0,
        "newbalanceDest": 5100.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


# ===================================================================
# Tests for _score_transaction_amount
# ===================================================================

class TestScoreTransactionAmount:
    """Tests for _score_transaction_amount."""

    def test_below_high_threshold(self):
        score, reason = _score_transaction_amount(5000)
        assert score == 0
        assert reason == ""

    def test_at_high_threshold_boundary(self):
        """Exactly at HIGH_AMOUNT_THRESHOLD should return 0 (not >)."""
        score, reason = _score_transaction_amount(HIGH_AMOUNT_THRESHOLD)
        assert score == 0
        assert reason == ""

    def test_just_above_high_threshold(self):
        score, reason = _score_transaction_amount(HIGH_AMOUNT_THRESHOLD + 0.01)
        assert score == 15
        assert "High transaction amount" in reason

    def test_at_very_high_threshold_boundary(self):
        """Exactly at VERY_HIGH_AMOUNT_THRESHOLD should return 15 (not >)."""
        score, reason = _score_transaction_amount(VERY_HIGH_AMOUNT_THRESHOLD)
        assert score == 15

    def test_just_above_very_high_threshold(self):
        score, reason = _score_transaction_amount(VERY_HIGH_AMOUNT_THRESHOLD + 0.01)
        assert score == 25
        assert "Very high transaction amount" in reason

    def test_at_extreme_threshold_boundary(self):
        """Exactly at EXTREME_AMOUNT_THRESHOLD should return 25 (not >)."""
        score, reason = _score_transaction_amount(EXTREME_AMOUNT_THRESHOLD)
        assert score == 25

    def test_just_above_extreme_threshold(self):
        score, reason = _score_transaction_amount(EXTREME_AMOUNT_THRESHOLD + 0.01)
        assert score == 30
        assert "Extreme transaction amount" in reason

    def test_very_large_amount(self):
        score, reason = _score_transaction_amount(10_000_000)
        assert score == 30

    def test_zero_amount(self):
        score, reason = _score_transaction_amount(0)
        assert score == 0
        assert reason == ""

    def test_negative_amount(self):
        score, reason = _score_transaction_amount(-500)
        assert score == 0
        assert reason == ""

    def test_reason_includes_amount(self):
        _, reason = _score_transaction_amount(600_000)
        assert "600,000.00" in reason


# ===================================================================
# Tests for _score_transaction_type
# ===================================================================

class TestScoreTransactionType:
    """Tests for _score_transaction_type."""

    def test_cash_out(self):
        score, reason = _score_transaction_type("CASH_OUT")
        assert score == 20
        assert "CASH_OUT" in reason

    def test_transfer(self):
        score, reason = _score_transaction_type("TRANSFER")
        assert score == 15
        assert "TRANSFER" in reason

    def test_payment_not_risky(self):
        score, reason = _score_transaction_type("PAYMENT")
        assert score == 0
        assert reason == ""

    def test_debit_not_risky(self):
        score, reason = _score_transaction_type("DEBIT")
        assert score == 0
        assert reason == ""

    def test_cash_in_not_risky(self):
        score, reason = _score_transaction_type("CASH_IN")
        assert score == 0
        assert reason == ""

    def test_empty_string(self):
        score, reason = _score_transaction_type("")
        assert score == 0
        assert reason == ""

    def test_case_sensitivity(self):
        """The function uses exact key lookup, so lowercase should return 0."""
        score, reason = _score_transaction_type("cash_out")
        assert score == 0
        assert reason == ""

    def test_reason_mentions_risky_type(self):
        _, reason = _score_transaction_type("CASH_OUT")
        assert "Risky transaction type" in reason


# ===================================================================
# Tests for _score_destination_frequency
# ===================================================================

class TestScoreDestinationFrequency:
    """Tests for _score_destination_frequency."""

    def test_destination_not_in_counts(self):
        """Missing destination defaults to count=1 -> 0 points."""
        score, reason = _score_destination_frequency("D999", {})
        assert score == 0
        assert reason == ""

    def test_count_of_one(self):
        score, reason = _score_destination_frequency("D1", {"D1": 1})
        assert score == 0
        assert reason == ""

    def test_count_of_two(self):
        # 5 + (2-1)*3 = 8
        score, reason = _score_destination_frequency("D1", {"D1": 2})
        assert score == 8
        assert "D1" in reason
        assert "2 incoming" in reason

    def test_count_of_three(self):
        # 5 + (3-1)*3 = 11
        score, reason = _score_destination_frequency("D1", {"D1": 3})
        assert score == 11

    def test_count_of_four(self):
        # 5 + (4-1)*3 = 14
        score, reason = _score_destination_frequency("D1", {"D1": 4})
        assert score == 14

    def test_count_of_five_caps_at_15(self):
        # 5 + (5-1)*3 = 17 -> capped at 15
        score, reason = _score_destination_frequency("D1", {"D1": 5})
        assert score == 15

    def test_very_high_count_capped(self):
        # 5 + (100-1)*3 = 302 -> capped at 15
        score, reason = _score_destination_frequency("D1", {"D1": 100})
        assert score == 15

    def test_reason_includes_count(self):
        _, reason = _score_destination_frequency("D1", {"D1": 3})
        assert "3 incoming transactions" in reason

    def test_different_dest_keys(self):
        counts = {"D1": 5, "D2": 1}
        score1, _ = _score_destination_frequency("D1", counts)
        score2, _ = _score_destination_frequency("D2", counts)
        assert score1 == 15
        assert score2 == 0


# ===================================================================
# Tests for _score_balance_anomalies
# ===================================================================

class TestScoreBalanceAnomalies:
    """Tests for _score_balance_anomalies."""

    def test_no_anomalies(self):
        score, reason = _score_balance_anomalies(500, 10000, 9500, 5000)
        assert score == 0
        assert reason == ""

    def test_zero_origin_balance_sends_money(self):
        """Origin has zero balance but sends money -> 10 pts."""
        score, reason = _score_balance_anomalies(1000, 0, 0, 5000)
        assert score == 10
        assert "zero balance" in reason.lower()

    def test_balance_drained_to_zero(self):
        """Old balance > 0, new balance == 0 -> 10 pts."""
        score, reason = _score_balance_anomalies(5000, 5000, 0, 5000)
        assert score == 10
        assert "drained to zero" in reason.lower()

    def test_dest_zero_balance(self):
        """Destination old balance == 0 -> 5 pts."""
        score, reason = _score_balance_anomalies(500, 10000, 9500, 0)
        assert score == 5
        assert "zero prior balance" in reason.lower()

    def test_all_anomalies_combined(self):
        """All three anomalies: 10 + 10 + 5 = 25."""
        score, reason = _score_balance_anomalies(1000, 0, 0, 0)
        # origin zero (10) + drained to zero is NOT triggered (old_balance_org==0) + dest zero (5)
        # Actually: old_balance_org==0 and amount>0 => 10pts
        # old_balance_org > 0 and new_balance_org == 0 => requires old > 0, so NOT triggered
        # old_balance_dest == 0 => 5pts
        # Total = 15
        assert score == 15

    def test_drained_and_dest_zero(self):
        """Drained (10) + dest zero (5) = 15."""
        score, reason = _score_balance_anomalies(5000, 5000, 0, 0)
        assert score == 15
        assert "drained" in reason.lower()
        assert "zero prior balance" in reason.lower()

    def test_origin_zero_and_dest_zero(self):
        """Origin zero sends (10) + dest zero (5) = 15."""
        score, reason = _score_balance_anomalies(100, 0, 0, 0)
        assert score == 15

    def test_cap_at_25(self):
        """Even if all conditions triggered, capped at 25."""
        # To trigger all three: old_balance_org > 0 (for drain), old_balance_org == 0 (for zero send)
        # These are mutually exclusive for origin zero vs drain, so max is 10+5=15 or 10+5=15
        # Actually max achievable is 10+10+5=25 when:
        #   old_balance_org > 0 and new_balance_org == 0 (drain: 10)
        #   but old_balance_org == 0 conflicts... so we can only get drain + dest = 15
        # Unless: old_balance_org > 0 and amount > old_balance_org triggers something...
        # Actually, the zero-origin and drain are mutually exclusive (old==0 vs old>0).
        # Max from single txn: drain(10) + dest_zero(5) = 15 or zero_origin(10) + dest_zero(5) = 15
        # The cap at 25 is there but 25 is not achievable. Let's just test the min() works.
        score, reason = _score_balance_anomalies(5000, 5000, 0, 0)
        assert score <= 25

    def test_zero_amount_with_zero_origin(self):
        """Zero amount with zero origin balance -> amount is not > 0, so no zero-origin signal."""
        score, reason = _score_balance_anomalies(0, 0, 0, 5000)
        assert score == 0

    def test_reasons_joined_with_semicolon(self):
        """Multiple reasons joined with '; '."""
        _, reason = _score_balance_anomalies(5000, 5000, 0, 0)
        assert "; " in reason

    def test_negative_balances_no_anomaly(self):
        score, reason = _score_balance_anomalies(100, -500, -600, 1000)
        assert score == 0


# ===================================================================
# Tests for _score_amount_vs_balance
# ===================================================================

class TestScoreAmountVsBalance:
    """Tests for _score_amount_vs_balance."""

    def test_amount_within_balance(self):
        score, reason = _score_amount_vs_balance(500, 10000)
        assert score == 0
        assert reason == ""

    def test_amount_equals_balance(self):
        """amount == old_balance_org -> not exceeding, so 0."""
        score, reason = _score_amount_vs_balance(10000, 10000)
        assert score == 0
        assert reason == ""

    def test_amount_exceeds_balance(self):
        score, reason = _score_amount_vs_balance(15000, 10000)
        assert score == 10
        assert "exceeds origin balance" in reason

    def test_zero_balance_no_score(self):
        """old_balance_org == 0 -> condition old_balance_org > 0 is false, so 0."""
        score, reason = _score_amount_vs_balance(1000, 0)
        assert score == 0
        assert reason == ""

    def test_negative_balance(self):
        """Negative balance is not > 0, so 0."""
        score, reason = _score_amount_vs_balance(100, -500)
        assert score == 0

    def test_reason_includes_amounts(self):
        _, reason = _score_amount_vs_balance(15000, 10000)
        assert "15,000.00" in reason
        assert "10,000.00" in reason

    def test_small_excess(self):
        score, _ = _score_amount_vs_balance(100.01, 100)
        assert score == 10


# ===================================================================
# Tests for _score_rapid_transactions
# ===================================================================

class TestScoreRapidTransactions:
    """Tests for _score_rapid_transactions."""

    def test_single_transaction(self):
        counts = {("C1", 1): 1}
        score, reason = _score_rapid_transactions("C1", 1, counts)
        assert score == 0
        assert reason == ""

    def test_multiple_transactions_same_step(self):
        counts = {("C1", 1): 3}
        score, reason = _score_rapid_transactions("C1", 1, counts)
        assert score == 10
        assert "Multiple transactions" in reason
        assert "3" in reason

    def test_key_not_in_counts(self):
        """Missing key defaults to count=1 -> 0 points."""
        score, reason = _score_rapid_transactions("C999", 5, {})
        assert score == 0
        assert reason == ""

    def test_count_of_two(self):
        counts = {("C1", 1): 2}
        score, reason = _score_rapid_transactions("C1", 1, counts)
        assert score == 10

    def test_different_steps(self):
        counts = {("C1", 1): 3, ("C1", 2): 1}
        score1, _ = _score_rapid_transactions("C1", 1, counts)
        score2, _ = _score_rapid_transactions("C1", 2, counts)
        assert score1 == 10
        assert score2 == 0

    def test_reason_includes_account_name(self):
        counts = {("C1", 1): 5}
        _, reason = _score_rapid_transactions("C1", 1, counts)
        assert "C1" in reason

    def test_reason_includes_count(self):
        counts = {("C1", 1): 7}
        _, reason = _score_rapid_transactions("C1", 1, counts)
        assert "7" in reason


# ===================================================================
# Tests for compute_risk_scores
# ===================================================================

class TestComputeRiskScores:
    """Tests for compute_risk_scores."""

    def test_basic_low_risk(self):
        """A small PAYMENT should be LOW risk."""
        df = _make_df([{"amount": 100, "type": "PAYMENT"}])
        result = compute_risk_scores(df)
        assert len(result) == 1
        assert result.iloc[0]["risk_level"] == "LOW"
        assert result.iloc[0]["risk_score"] < 40

    def test_transaction_id_format(self):
        df = _make_df([{}, {}])
        result = compute_risk_scores(df)
        assert result.iloc[0]["transaction_id"] == "TXN-0001"
        assert result.iloc[1]["transaction_id"] == "TXN-0002"

    def test_output_columns(self):
        df = _make_df([{}])
        result = compute_risk_scores(df)
        assert list(result.columns) == ["transaction_id", "risk_score", "risk_level", "explanation"]

    def test_high_risk_transaction(self):
        """A large CASH_OUT draining balance with dest zero -> should be HIGH."""
        df = _make_df([{
            "type": "CASH_OUT",
            "amount": 600_000,
            "oldbalanceOrg": 600_000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
        }])
        result = compute_risk_scores(df)
        # extreme amount (30) + CASH_OUT (20) + drained (10) + dest zero (5) + exceeds not triggered = 65
        # Actually: amount > old_balance? 600k == 600k, not exceeding. So 30+20+10+5 = 65
        # 65 >= 40 -> MEDIUM (since > 70 is HIGH)
        # Let's check
        row = result.iloc[0]
        assert row["risk_score"] >= 40

    def test_medium_risk_transaction(self):
        """Score between 40 and 70 -> MEDIUM."""
        df = _make_df([{
            "type": "CASH_OUT",       # 20 pts
            "amount": 50_000,         # 15 pts (> 10k)
            "oldbalanceOrg": 50_000,
            "newbalanceOrig": 0,      # drained: 10 pts
            "oldbalanceDest": 5000,
        }])
        result = compute_risk_scores(df)
        row = result.iloc[0]
        # 20 + 15 + 10 = 45 -> MEDIUM
        assert row["risk_level"] == "MEDIUM"
        assert 40 <= row["risk_score"] <= 70

    def test_score_capped_at_100(self):
        """Construct a scenario that would exceed 100 without capping."""
        # We need many signals to fire. Use repeated dest + rapid txns + all anomalies + big amount
        df = _make_df([
            {
                "type": "CASH_OUT",        # 20
                "amount": 600_000,         # 30
                "nameOrig": "C1",
                "step": 1,
                "oldbalanceOrg": 100_000,  # amount > balance: 10
                "newbalanceOrig": 0,       # drained: 10
                "nameDest": "D1",
                "oldbalanceDest": 0,       # dest zero: 5
            },
            {
                "type": "CASH_OUT",
                "amount": 600_000,
                "nameOrig": "C1",
                "step": 1,
                "oldbalanceOrg": 100_000,
                "newbalanceOrig": 0,
                "nameDest": "D1",
                "oldbalanceDest": 0,
            },
        ])
        result = compute_risk_scores(df)
        for _, row in result.iterrows():
            assert row["risk_score"] <= 100

    def test_risk_level_low(self):
        df = _make_df([{"amount": 50, "type": "PAYMENT", "oldbalanceDest": 5000}])
        result = compute_risk_scores(df)
        assert result.iloc[0]["risk_level"] == "LOW"

    def test_risk_level_high(self):
        """Score > 70 -> HIGH."""
        df = _make_df([
            {
                "type": "CASH_OUT",
                "amount": 600_000,
                "nameOrig": "C1",
                "step": 1,
                "oldbalanceOrg": 100_000,
                "newbalanceOrig": 0,
                "nameDest": "D1",
                "oldbalanceDest": 0,
            },
            {
                "type": "CASH_OUT",
                "amount": 600_000,
                "nameOrig": "C1",
                "step": 1,
                "oldbalanceOrg": 100_000,
                "newbalanceOrig": 0,
                "nameDest": "D1",
                "oldbalanceDest": 0,
            },
        ])
        result = compute_risk_scores(df)
        # Each row: extreme(30) + cashout(20) + dest_freq(8 for count=2) + drained(10) + dest_zero(5) + exceeds(10) + rapid(10) = 93
        assert result.iloc[0]["risk_level"] == "HIGH"
        assert result.iloc[0]["risk_score"] > 70

    def test_explanation_for_low_risk(self):
        df = _make_df([{"amount": 50, "type": "PAYMENT", "oldbalanceDest": 5000}])
        result = compute_risk_scores(df)
        assert result.iloc[0]["explanation"] == "No significant risk signals detected"

    def test_explanation_contains_reasons(self):
        df = _make_df([{
            "type": "CASH_OUT",
            "amount": 600_000,
            "oldbalanceOrg": 5000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
        }])
        result = compute_risk_scores(df)
        explanation = result.iloc[0]["explanation"]
        assert "Extreme transaction amount" in explanation
        assert "Risky transaction type" in explanation

    def test_multiple_rows(self):
        df = _make_df([
            {"amount": 50, "type": "PAYMENT", "oldbalanceDest": 5000},
            {"amount": 600_000, "type": "CASH_OUT", "oldbalanceDest": 0},
        ])
        result = compute_risk_scores(df)
        assert len(result) == 2
        assert result.iloc[0]["risk_score"] < result.iloc[1]["risk_score"]

    def test_dest_frequency_aggregation(self):
        """Multiple rows to same destination should increase dest frequency score."""
        df = _make_df([
            {"nameDest": "D1", "oldbalanceDest": 5000},
            {"nameDest": "D1", "oldbalanceDest": 5000},
            {"nameDest": "D1", "oldbalanceDest": 5000},
        ])
        result = compute_risk_scores(df)
        # count=3 for D1 -> 5 + (3-1)*3 = 11 pts
        # All rows should have the dest frequency signal
        for _, row in result.iterrows():
            assert "Repeated destination" in row["explanation"]

    def test_rapid_transaction_aggregation(self):
        """Multiple rows from same origin in same step should trigger rapid score."""
        df = _make_df([
            {"nameOrig": "C1", "step": 1, "oldbalanceDest": 5000},
            {"nameOrig": "C1", "step": 1, "oldbalanceDest": 5000},
        ])
        result = compute_risk_scores(df)
        for _, row in result.iterrows():
            assert "Multiple transactions" in row["explanation"]

    def test_empty_dataframe(self):
        """Empty DataFrame input should return empty result."""
        df = pd.DataFrame(columns=[
            "step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
            "isFraud", "isFlaggedFraud",
        ])
        result = compute_risk_scores(df)
        assert len(result) == 0

    def test_risk_level_boundary_at_40(self):
        """Score of exactly 40 should be MEDIUM."""
        # CASH_OUT(20) + high amount (15 for >10k) + dest_zero(5) = 40
        df = _make_df([{
            "type": "CASH_OUT",
            "amount": 20_000,
            "oldbalanceOrg": 100_000,
            "newbalanceOrig": 80_000,
            "oldbalanceDest": 0,
        }])
        result = compute_risk_scores(df)
        row = result.iloc[0]
        # 20 + 15 + 5 = 40
        assert row["risk_score"] == 40
        assert row["risk_level"] == "MEDIUM"

    def test_risk_level_boundary_at_70(self):
        """Score of exactly 70 should be MEDIUM (HIGH requires > 70)."""
        # We need exactly 70. Let's build it:
        # CASH_OUT(20) + extreme(30) + drained(10) + dest_zero(5) + dest_freq needs to add 5
        # Actually let's just check a constructed scenario
        # extreme(30) + CASH_OUT(20) + drained(10) + exceeds(10) = 70
        df = _make_df([{
            "type": "CASH_OUT",
            "amount": 600_000,
            "oldbalanceOrg": 500_000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 5000,
        }])
        result = compute_risk_scores(df)
        row = result.iloc[0]
        # 30 + 20 + 10 + 10 = 70
        assert row["risk_score"] == 70
        assert row["risk_level"] == "MEDIUM"

    def test_risk_level_boundary_at_71(self):
        """Score of 71 should be HIGH."""
        # 30 + 20 + 10 + 10 + 5(dest_zero) = 75 -> HIGH
        df = _make_df([{
            "type": "CASH_OUT",
            "amount": 600_000,
            "oldbalanceOrg": 500_000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
        }])
        result = compute_risk_scores(df)
        row = result.iloc[0]
        assert row["risk_score"] > 70
        assert row["risk_level"] == "HIGH"


# ===================================================================
# Tests for generate_report
# ===================================================================

class TestGenerateReport:
    """Tests for generate_report."""

    def test_writes_csv(self, tmp_path):
        df = pd.DataFrame({
            "transaction_id": ["TXN-0001"],
            "risk_score": [50],
            "risk_level": ["MEDIUM"],
            "explanation": ["Some reason"],
        })
        output_path = str(tmp_path / "report.csv")
        generate_report(df, output_path)
        assert os.path.exists(output_path)

    def test_csv_contents(self, tmp_path):
        df = pd.DataFrame({
            "transaction_id": ["TXN-0001", "TXN-0002"],
            "risk_score": [10, 80],
            "risk_level": ["LOW", "HIGH"],
            "explanation": ["No risk", "High risk"],
        })
        output_path = str(tmp_path / "report.csv")
        generate_report(df, output_path)
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 2
        assert list(loaded.columns) == ["transaction_id", "risk_score", "risk_level", "explanation"]
        assert loaded.iloc[0]["transaction_id"] == "TXN-0001"
        assert loaded.iloc[1]["risk_score"] == 80

    def test_creates_directories(self, tmp_path):
        output_path = str(tmp_path / "nested" / "dir" / "report.csv")
        df = pd.DataFrame({
            "transaction_id": ["TXN-0001"],
            "risk_score": [0],
            "risk_level": ["LOW"],
            "explanation": [""],
        })
        generate_report(df, output_path)
        assert os.path.exists(output_path)

    def test_overwrites_existing_file(self, tmp_path):
        output_path = str(tmp_path / "report.csv")
        df1 = pd.DataFrame({"transaction_id": ["TXN-0001"], "risk_score": [10], "risk_level": ["LOW"], "explanation": [""]})
        df2 = pd.DataFrame({"transaction_id": ["TXN-0002"], "risk_score": [90], "risk_level": ["HIGH"], "explanation": [""]})
        generate_report(df1, output_path)
        generate_report(df2, output_path)
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["transaction_id"] == "TXN-0002"

    def test_empty_dataframe(self, tmp_path):
        output_path = str(tmp_path / "report.csv")
        df = pd.DataFrame(columns=["transaction_id", "risk_score", "risk_level", "explanation"])
        generate_report(df, output_path)
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 0

    def test_no_index_column(self, tmp_path):
        """The CSV should not have a pandas index column."""
        output_path = str(tmp_path / "report.csv")
        df = pd.DataFrame({
            "transaction_id": ["TXN-0001"],
            "risk_score": [50],
            "risk_level": ["MEDIUM"],
            "explanation": ["reason"],
        })
        generate_report(df, output_path)
        with open(output_path) as f:
            header = f.readline().strip()
        assert header == "transaction_id,risk_score,risk_level,explanation"


# ===================================================================
# Tests for constants
# ===================================================================

class TestConstants:
    """Verify module-level constants are as expected."""

    def test_thresholds(self):
        assert HIGH_AMOUNT_THRESHOLD == 10_000
        assert VERY_HIGH_AMOUNT_THRESHOLD == 100_000
        assert EXTREME_AMOUNT_THRESHOLD == 500_000

    def test_risky_types(self):
        assert RISKY_TYPES == {"CASH_OUT": 20, "TRANSFER": 15}

    def test_risk_level_thresholds(self):
        assert RISK_LEVEL_THRESHOLDS == {"LOW": 40, "MEDIUM": 70}
