"""
Comprehensive unit tests for the fraud_risk_scoring module.

Covers all scoring functions, risk level assignment, explanation building,
the main compute pipeline, report generation, and summary printing.
"""

import os

import pandas as pd
import pytest

from fraud_risk_scoring import (
    HIGH_AMOUNT_THRESHOLD,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    WEIGHT_AMOUNT,
    WEIGHT_AMOUNT_VS_BALANCE,
    WEIGHT_BALANCE_DRAIN,
    WEIGHT_DEST_BALANCE_DROP,
    WEIGHT_DEST_FREQUENCY,
    WEIGHT_TYPE,
    WEIGHT_ZERO_ORIG_BALANCE,
    assign_risk_level,
    build_explanation,
    compute_risk_scores,
    generate_report,
    print_summary,
    score_amount,
    score_amount_vs_balance,
    score_balance_drain,
    score_dest_balance_drop,
    score_dest_frequency,
    score_type,
    score_zero_orig_balance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**overrides) -> pd.Series:
    """Create a transaction row (pd.Series) with sensible defaults."""
    defaults = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C1000",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9900.0,
        "nameDest": "M1000",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of row dicts, filling missing columns."""
    base = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C1000",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9900.0,
        "nameDest": "M1000",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    full_rows = []
    for r in rows:
        row = {**base, **r}
        full_rows.append(row)
    return pd.DataFrame(full_rows)


# ===================================================================
# score_amount
# ===================================================================


class TestScoreAmount:
    """Tests for score_amount(amount)."""

    def test_above_500k(self):
        assert score_amount(500001) == WEIGHT_AMOUNT  # 25

    def test_exactly_500k(self):
        # 500000 is NOT > 500000
        assert score_amount(500000) == WEIGHT_AMOUNT * 0.8

    def test_above_100k(self):
        assert score_amount(200000) == WEIGHT_AMOUNT * 0.8  # 20

    def test_exactly_100k(self):
        # 100000 is NOT > 100000; falls to next bracket > 10000
        assert score_amount(100000) == WEIGHT_AMOUNT * 0.5

    def test_above_10k(self):
        assert score_amount(10001) == WEIGHT_AMOUNT * 0.5  # 12.5

    def test_exactly_10k(self):
        # 10000 is NOT > 10000; falls to > 5000
        assert score_amount(10000) == WEIGHT_AMOUNT * 0.2

    def test_above_5k(self):
        assert score_amount(5001) == WEIGHT_AMOUNT * 0.2  # 5

    def test_exactly_5k(self):
        # 5000 is NOT > 5000
        assert score_amount(5000) == 0

    def test_below_5k(self):
        assert score_amount(4999) == 0

    def test_zero(self):
        assert score_amount(0) == 0

    def test_small_amount(self):
        assert score_amount(1) == 0

    def test_negative_amount(self):
        assert score_amount(-100) == 0

    def test_very_large_amount(self):
        assert score_amount(10_000_000) == WEIGHT_AMOUNT


# ===================================================================
# score_type
# ===================================================================


class TestScoreType:
    """Tests for score_type(txn_type)."""

    def test_cash_out(self):
        assert score_type("CASH_OUT") == WEIGHT_TYPE  # 20

    def test_transfer(self):
        assert score_type("TRANSFER") == WEIGHT_TYPE  # 20

    def test_debit(self):
        assert score_type("DEBIT") == WEIGHT_TYPE * 0.3  # 6

    def test_payment(self):
        assert score_type("PAYMENT") == 0

    def test_cash_in(self):
        assert score_type("CASH_IN") == 0

    def test_unknown_type(self):
        assert score_type("UNKNOWN") == 0

    def test_empty_string(self):
        assert score_type("") == 0

    def test_lowercase_not_matched(self):
        assert score_type("cash_out") == 0


# ===================================================================
# score_balance_drain
# ===================================================================


class TestScoreBalanceDrain:
    """Tests for score_balance_drain(old_balance, new_balance)."""

    def test_fully_drained(self):
        assert score_balance_drain(10000, 0) == WEIGHT_BALANCE_DRAIN  # 15

    def test_nearly_drained(self):
        # new_balance < old * 0.1  =>  999 < 10000*0.1=1000
        assert score_balance_drain(10000, 999) == WEIGHT_BALANCE_DRAIN * 0.5  # 7.5

    def test_exactly_10_percent(self):
        # new_balance == old * 0.1 => 1000 is NOT < 1000
        assert score_balance_drain(10000, 1000) == 0

    def test_above_10_percent(self):
        assert score_balance_drain(10000, 5000) == 0

    def test_old_zero_new_zero(self):
        # old == 0, condition old > 0 fails
        assert score_balance_drain(0, 0) == 0

    def test_old_zero_new_positive(self):
        assert score_balance_drain(0, 100) == 0

    def test_old_positive_new_negative(self):
        # new_balance < old * 0.1 => True
        assert score_balance_drain(100, -10) == WEIGHT_BALANCE_DRAIN * 0.5

    def test_small_balance_drained(self):
        assert score_balance_drain(1, 0) == WEIGHT_BALANCE_DRAIN


# ===================================================================
# score_dest_frequency
# ===================================================================


class TestScoreDestFrequency:
    """Tests for score_dest_frequency(dest_account, dest_counts)."""

    def test_count_5_or_more(self):
        counts = {"D1": 5}
        assert score_dest_frequency("D1", counts) == WEIGHT_DEST_FREQUENCY  # 15

    def test_count_above_5(self):
        counts = {"D1": 10}
        assert score_dest_frequency("D1", counts) == WEIGHT_DEST_FREQUENCY

    def test_count_3(self):
        counts = {"D1": 3}
        assert score_dest_frequency("D1", counts) == WEIGHT_DEST_FREQUENCY * 0.6  # 9

    def test_count_4(self):
        counts = {"D1": 4}
        assert score_dest_frequency("D1", counts) == WEIGHT_DEST_FREQUENCY * 0.6

    def test_count_2(self):
        counts = {"D1": 2}
        assert score_dest_frequency("D1", counts) == WEIGHT_DEST_FREQUENCY * 0.3  # 4.5

    def test_count_1(self):
        counts = {"D1": 1}
        assert score_dest_frequency("D1", counts) == 0

    def test_missing_key_defaults_to_1(self):
        # dest_counts.get returns 1 if key missing
        assert score_dest_frequency("UNKNOWN", {}) == 0

    def test_count_0(self):
        counts = {"D1": 0}
        assert score_dest_frequency("D1", counts) == 0


# ===================================================================
# score_zero_orig_balance
# ===================================================================


class TestScoreZeroOrigBalance:
    """Tests for score_zero_orig_balance(old_balance)."""

    def test_zero_balance(self):
        assert score_zero_orig_balance(0) == WEIGHT_ZERO_ORIG_BALANCE  # 10

    def test_positive_balance(self):
        assert score_zero_orig_balance(1) == 0

    def test_negative_balance(self):
        assert score_zero_orig_balance(-100) == 0

    def test_large_balance(self):
        assert score_zero_orig_balance(999999) == 0

    def test_float_zero(self):
        assert score_zero_orig_balance(0.0) == WEIGHT_ZERO_ORIG_BALANCE


# ===================================================================
# score_dest_balance_drop
# ===================================================================


class TestScoreDestBalanceDrop:
    """Tests for score_dest_balance_drop(old_dest_balance, new_dest_balance)."""

    def test_drop_to_zero(self):
        assert score_dest_balance_drop(1000, 0) == WEIGHT_DEST_BALANCE_DROP  # 10

    def test_no_drop(self):
        assert score_dest_balance_drop(1000, 500) == 0

    def test_old_zero(self):
        assert score_dest_balance_drop(0, 0) == 0

    def test_old_zero_new_positive(self):
        assert score_dest_balance_drop(0, 100) == 0

    def test_both_positive(self):
        assert score_dest_balance_drop(100, 100) == 0

    def test_old_positive_new_negative(self):
        # new != 0
        assert score_dest_balance_drop(100, -50) == 0

    def test_small_old_drop(self):
        assert score_dest_balance_drop(0.01, 0) == WEIGHT_DEST_BALANCE_DROP


# ===================================================================
# score_amount_vs_balance
# ===================================================================


class TestScoreAmountVsBalance:
    """Tests for score_amount_vs_balance(amount, old_balance)."""

    def test_amount_equals_balance(self):
        assert score_amount_vs_balance(10000, 10000) == WEIGHT_AMOUNT_VS_BALANCE  # 5

    def test_amount_exceeds_balance(self):
        assert score_amount_vs_balance(20000, 10000) == WEIGHT_AMOUNT_VS_BALANCE

    def test_amount_less_than_balance(self):
        assert score_amount_vs_balance(5000, 10000) == 0

    def test_zero_balance(self):
        # old_balance == 0, condition old > 0 fails
        assert score_amount_vs_balance(100, 0) == 0

    def test_zero_amount_zero_balance(self):
        assert score_amount_vs_balance(0, 0) == 0

    def test_zero_amount_positive_balance(self):
        # 0 < 100 => not >=
        assert score_amount_vs_balance(0, 100) == 0


# ===================================================================
# assign_risk_level
# ===================================================================


class TestAssignRiskLevel:
    """Tests for assign_risk_level(score)."""

    def test_high_above_70(self):
        assert assign_risk_level(71) == "HIGH"

    def test_exactly_70(self):
        # 70 is NOT > 70
        assert assign_risk_level(70) == "MEDIUM"

    def test_medium_at_40(self):
        assert assign_risk_level(40) == "MEDIUM"

    def test_medium_between_40_and_70(self):
        assert assign_risk_level(55) == "MEDIUM"

    def test_low_below_40(self):
        assert assign_risk_level(39) == "LOW"

    def test_low_zero(self):
        assert assign_risk_level(0) == "LOW"

    def test_high_100(self):
        assert assign_risk_level(100) == "HIGH"

    def test_negative_score(self):
        assert assign_risk_level(-10) == "LOW"

    def test_boundary_just_above_70(self):
        assert assign_risk_level(70.01) == "HIGH"

    def test_boundary_just_below_40(self):
        assert assign_risk_level(39.99) == "LOW"


# ===================================================================
# build_explanation
# ===================================================================


class TestBuildExplanation:
    """Tests for build_explanation(row, dest_counts)."""

    def test_high_amount_signal(self):
        row = _make_row(amount=50000)
        explanation = build_explanation(row, {})
        assert "High transaction amount" in explanation
        assert "$50,000.00" in explanation

    def test_risky_type_cash_out(self):
        row = _make_row(type="CASH_OUT")
        explanation = build_explanation(row, {})
        assert "Risky transaction type (CASH_OUT)" in explanation

    def test_risky_type_transfer(self):
        row = _make_row(type="TRANSFER")
        explanation = build_explanation(row, {})
        assert "Risky transaction type (TRANSFER)" in explanation

    def test_non_risky_type_no_signal(self):
        row = _make_row(type="PAYMENT", amount=100)
        explanation = build_explanation(row, {})
        assert "Risky transaction type" not in explanation

    def test_fully_drained(self):
        row = _make_row(oldbalanceOrg=5000, newbalanceOrig=0)
        explanation = build_explanation(row, {})
        assert "Account fully drained" in explanation

    def test_nearly_drained(self):
        row = _make_row(oldbalanceOrg=10000, newbalanceOrig=500)
        explanation = build_explanation(row, {})
        assert "Account nearly drained" in explanation

    def test_not_drained(self):
        row = _make_row(oldbalanceOrg=10000, newbalanceOrig=5000)
        explanation = build_explanation(row, {})
        assert "drained" not in explanation

    def test_dest_frequency_signal(self):
        row = _make_row(nameDest="D1")
        counts = {"D1": 3}
        explanation = build_explanation(row, counts)
        assert "Destination account (D1) received 3 transactions" in explanation

    def test_dest_frequency_below_2(self):
        row = _make_row(nameDest="D1")
        counts = {"D1": 1}
        explanation = build_explanation(row, counts)
        assert "Destination account" not in explanation

    def test_zero_orig_balance_signal(self):
        row = _make_row(oldbalanceOrg=0, newbalanceOrig=0)
        explanation = build_explanation(row, {})
        assert "Originating account had zero balance" in explanation

    def test_dest_balance_drop_signal(self):
        row = _make_row(oldbalanceDest=500, newbalanceDest=0)
        explanation = build_explanation(row, {})
        assert "Destination account balance dropped to zero" in explanation

    def test_amount_vs_balance_signal(self):
        row = _make_row(amount=10000, oldbalanceOrg=10000, newbalanceOrig=0)
        explanation = build_explanation(row, {})
        assert "Transaction amount equals or exceeds full account balance" in explanation

    def test_no_signals(self):
        row = _make_row(
            type="PAYMENT",
            amount=100,
            oldbalanceOrg=10000,
            newbalanceOrig=9900,
            oldbalanceDest=0,
            newbalanceDest=100,
        )
        explanation = build_explanation(row, {})
        assert explanation == "No significant risk signals detected"

    def test_multiple_signals_joined_by_semicolons(self):
        row = _make_row(
            type="CASH_OUT",
            amount=50000,
            oldbalanceOrg=50000,
            newbalanceOrig=0,
            oldbalanceDest=100,
            newbalanceDest=0,
            nameDest="D1",
        )
        counts = {"D1": 5}
        explanation = build_explanation(row, counts)
        assert "; " in explanation
        assert "High transaction amount" in explanation
        assert "Risky transaction type" in explanation
        assert "Account fully drained" in explanation
        assert "Destination account (D1) received 5 transactions" in explanation
        assert "Destination account balance dropped to zero" in explanation
        assert "Transaction amount equals or exceeds full account balance" in explanation


# ===================================================================
# compute_risk_scores
# ===================================================================


class TestComputeRiskScores:
    """Tests for compute_risk_scores(df)."""

    def test_basic_low_risk_transaction(self):
        df = _make_df([{
            "type": "PAYMENT",
            "amount": 100,
            "nameOrig": "C1",
            "oldbalanceOrg": 10000,
            "newbalanceOrig": 9900,
            "nameDest": "M1",
            "oldbalanceDest": 0,
            "newbalanceDest": 100,
        }])
        result = compute_risk_scores(df)
        assert len(result) == 1
        assert result.iloc[0]["risk_level"] == "LOW"
        assert result.iloc[0]["risk_score"] == 0

    def test_high_risk_transaction(self):
        df = _make_df([{
            "type": "CASH_OUT",
            "amount": 600000,
            "nameOrig": "C1",
            "oldbalanceOrg": 600000,
            "newbalanceOrig": 0,
            "nameDest": "D1",
            "oldbalanceDest": 500,
            "newbalanceDest": 0,
        }])
        result = compute_risk_scores(df)
        row = result.iloc[0]
        assert row["risk_level"] == "HIGH"
        # WEIGHT_AMOUNT(25) + WEIGHT_TYPE(20) + WEIGHT_BALANCE_DRAIN(15)
        # + WEIGHT_DEST_BALANCE_DROP(10) + WEIGHT_AMOUNT_VS_BALANCE(5) = 75
        assert row["risk_score"] == 75

    def test_output_columns(self):
        df = _make_df([{}])
        result = compute_risk_scores(df)
        expected_cols = [
            "transaction_id", "type", "amount", "nameOrig", "nameDest",
            "risk_score", "risk_level", "explanation",
        ]
        assert list(result.columns) == expected_cols

    def test_transaction_id_starts_at_1(self):
        df = _make_df([{}, {}, {}])
        result = compute_risk_scores(df)
        assert list(result["transaction_id"]) == [1, 2, 3]

    def test_score_capped_at_100(self):
        """Construct a scenario where raw score would exceed 100."""
        # zero orig balance (10) + CASH_OUT (20) + amount>500k (25)
        # + balance drain (15) + dest freq (we'll add 5 identical dests) (15)
        # + dest balance drop (10) + amount vs balance (5) = 100
        # All signals active => 100 cap
        rows = []
        for i in range(5):
            rows.append({
                "type": "CASH_OUT",
                "amount": 600000,
                "nameOrig": f"C{i}",
                "oldbalanceOrg": 0,
                "newbalanceOrig": 0,
                "nameDest": "D_COMMON",
                "oldbalanceDest": 1000,
                "newbalanceDest": 0,
            })
        df = _make_df(rows)
        result = compute_risk_scores(df)
        for _, row in result.iterrows():
            assert row["risk_score"] <= 100

    def test_dest_frequency_across_transactions(self):
        """Multiple transactions to the same dest should increase dest freq score."""
        rows = [
            {"nameDest": "D1", "type": "PAYMENT", "amount": 100,
             "oldbalanceOrg": 10000, "newbalanceOrig": 9900,
             "oldbalanceDest": 0, "newbalanceDest": 100},
            {"nameDest": "D1", "type": "PAYMENT", "amount": 100,
             "oldbalanceOrg": 10000, "newbalanceOrig": 9900,
             "oldbalanceDest": 0, "newbalanceDest": 100},
            {"nameDest": "D1", "type": "PAYMENT", "amount": 100,
             "oldbalanceOrg": 10000, "newbalanceOrig": 9900,
             "oldbalanceDest": 0, "newbalanceDest": 100},
        ]
        df = _make_df(rows)
        result = compute_risk_scores(df)
        # dest_count for D1 = 3, so score_dest_frequency = 15 * 0.6 = 9
        for _, row in result.iterrows():
            assert row["risk_score"] == 9.0

    def test_medium_risk_transaction(self):
        """A transaction that lands in MEDIUM risk."""
        df = _make_df([{
            "type": "TRANSFER",
            "amount": 200000,
            "nameOrig": "C1",
            "oldbalanceOrg": 200000,
            "newbalanceOrig": 0,
            "nameDest": "D1",
            "oldbalanceDest": 0,
            "newbalanceDest": 200000,
        }])
        result = compute_risk_scores(df)
        row = result.iloc[0]
        # WEIGHT_AMOUNT*0.8(20) + WEIGHT_TYPE(20) + WEIGHT_BALANCE_DRAIN(15)
        # + WEIGHT_AMOUNT_VS_BALANCE(5) = 60
        assert row["risk_score"] == 60
        assert row["risk_level"] == "MEDIUM"

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty result."""
        df = pd.DataFrame(columns=[
            "step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
            "isFraud", "isFlaggedFraud",
        ])
        result = compute_risk_scores(df)
        assert len(result) == 0


# ===================================================================
# generate_report
# ===================================================================


class TestGenerateReport:
    """Tests for generate_report(df_scores, output_path)."""

    def test_creates_csv(self, tmp_path):
        df = pd.DataFrame({
            "transaction_id": [1],
            "type": ["PAYMENT"],
            "amount": [100],
            "nameOrig": ["C1"],
            "nameDest": ["M1"],
            "risk_score": [0],
            "risk_level": ["LOW"],
            "explanation": ["No significant risk signals detected"],
        })
        output = str(tmp_path / "out" / "report.csv")
        generate_report(df, output)
        assert os.path.exists(output)
        loaded = pd.read_csv(output)
        assert len(loaded) == 1
        assert loaded.iloc[0]["risk_level"] == "LOW"

    def test_creates_nested_directories(self, tmp_path):
        output = str(tmp_path / "a" / "b" / "c" / "report.csv")
        df = pd.DataFrame({"col": [1]})
        generate_report(df, output)
        assert os.path.exists(output)

    def test_prints_confirmation(self, tmp_path, capsys):
        output = str(tmp_path / "report.csv")
        df = pd.DataFrame({"col": [1]})
        generate_report(df, output)
        captured = capsys.readouterr()
        assert f"Report saved to {output}" in captured.out

    def test_overwrites_existing_file(self, tmp_path):
        output = str(tmp_path / "report.csv")
        df1 = pd.DataFrame({"col": [1]})
        df2 = pd.DataFrame({"col": [2, 3]})
        generate_report(df1, output)
        generate_report(df2, output)
        loaded = pd.read_csv(output)
        assert len(loaded) == 2


# ===================================================================
# print_summary
# ===================================================================


class TestPrintSummary:
    """Tests for print_summary(df_scores)."""

    def test_summary_output(self, capsys):
        df = pd.DataFrame({
            "transaction_id": [1, 2, 3],
            "type": ["CASH_OUT", "TRANSFER", "PAYMENT"],
            "amount": [600000, 50000, 100],
            "nameOrig": ["C1", "C2", "C3"],
            "nameDest": ["D1", "D2", "D3"],
            "risk_score": [80, 50, 10],
            "risk_level": ["HIGH", "MEDIUM", "LOW"],
            "explanation": ["sig1", "sig2", "sig3"],
        })
        print_summary(df)
        captured = capsys.readouterr().out
        assert "FRAUD RISK SCORING SUMMARY" in captured
        assert "Total transactions analyzed: 3" in captured
        assert "HIGH risk:" in captured
        assert "MEDIUM risk:" in captured
        assert "LOW risk:" in captured

    def test_summary_percentages(self, capsys):
        df = pd.DataFrame({
            "transaction_id": [1, 2],
            "type": ["CASH_OUT", "PAYMENT"],
            "amount": [600000, 100],
            "nameOrig": ["C1", "C2"],
            "nameDest": ["D1", "D2"],
            "risk_score": [80, 10],
            "risk_level": ["HIGH", "LOW"],
            "explanation": ["sig1", "sig2"],
        })
        print_summary(df)
        captured = capsys.readouterr().out
        assert "50.0%" in captured

    def test_top_high_risk_displayed(self, capsys):
        df = pd.DataFrame({
            "transaction_id": [1],
            "type": ["CASH_OUT"],
            "amount": [600000],
            "nameOrig": ["C1"],
            "nameDest": ["D1"],
            "risk_score": [80],
            "risk_level": ["HIGH"],
            "explanation": ["High risk signals"],
        })
        print_summary(df)
        captured = capsys.readouterr().out
        assert "Top HIGH risk transactions:" in captured
        assert "TX #  1" in captured
        assert "Score:    80" in captured

    def test_no_high_risk(self, capsys):
        df = pd.DataFrame({
            "transaction_id": [1],
            "type": ["PAYMENT"],
            "amount": [100],
            "nameOrig": ["C1"],
            "nameDest": ["M1"],
            "risk_score": [0],
            "risk_level": ["LOW"],
            "explanation": ["No signals"],
        })
        print_summary(df)
        captured = capsys.readouterr().out
        assert "Top HIGH risk transactions:" not in captured

    def test_all_risk_levels_present(self, capsys):
        df = pd.DataFrame({
            "transaction_id": [1, 2, 3, 4],
            "type": ["CASH_OUT", "TRANSFER", "PAYMENT", "PAYMENT"],
            "amount": [600000, 50000, 100, 200],
            "nameOrig": ["C1", "C2", "C3", "C4"],
            "nameDest": ["D1", "D2", "D3", "D4"],
            "risk_score": [90, 50, 10, 5],
            "risk_level": ["HIGH", "MEDIUM", "LOW", "LOW"],
            "explanation": ["s1", "s2", "s3", "s4"],
        })
        print_summary(df)
        captured = capsys.readouterr().out
        assert "HIGH risk:     1 (25.0%)" in captured
        assert "MEDIUM risk:   1 (25.0%)" in captured
        assert "LOW risk:      2 (50.0%)" in captured


# ===================================================================
# Integration: constants sanity checks
# ===================================================================


class TestConstants:
    """Verify module-level constants are as expected."""

    def test_high_amount_threshold(self):
        assert HIGH_AMOUNT_THRESHOLD == 10000

    def test_low_threshold(self):
        assert LOW_THRESHOLD == 40

    def test_high_threshold(self):
        assert HIGH_THRESHOLD == 70

    def test_weight_amount(self):
        assert WEIGHT_AMOUNT == 25

    def test_weight_type(self):
        assert WEIGHT_TYPE == 20

    def test_weight_balance_drain(self):
        assert WEIGHT_BALANCE_DRAIN == 15

    def test_weight_dest_frequency(self):
        assert WEIGHT_DEST_FREQUENCY == 15

    def test_weight_zero_orig_balance(self):
        assert WEIGHT_ZERO_ORIG_BALANCE == 10

    def test_weight_dest_balance_drop(self):
        assert WEIGHT_DEST_BALANCE_DROP == 10

    def test_weight_amount_vs_balance(self):
        assert WEIGHT_AMOUNT_VS_BALANCE == 5
