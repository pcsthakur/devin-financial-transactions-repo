package com.anomaly.model;

import java.util.List;

/**
 * Represents a detected anomaly for a customer, including the pattern type,
 * affected transactions, risk score, and a human-readable explanation.
 */
public class Anomaly {

    private final String customer;
    private final AnomalyType type;
    private final int riskScore;
    private final String riskLevel;
    private final String explanation;
    private final List<Integer> transactionRows;

    public Anomaly(String customer, AnomalyType type, int riskScore,
                   String explanation, List<Integer> transactionRows) {
        this.customer = customer;
        this.type = type;
        this.riskScore = Math.min(riskScore, 100);
        this.riskLevel = assignRiskLevel(this.riskScore);
        this.explanation = explanation;
        this.transactionRows = transactionRows;
    }

    private static String assignRiskLevel(int score) {
        if (score > 70) {
            return "HIGH";
        } else if (score >= 40) {
            return "MEDIUM";
        }
        return "LOW";
    }

    public String getCustomer() {
        return customer;
    }

    public AnomalyType getType() {
        return type;
    }

    public int getRiskScore() {
        return riskScore;
    }

    public String getRiskLevel() {
        return riskLevel;
    }

    public String getExplanation() {
        return explanation;
    }

    public List<Integer> getTransactionRows() {
        return transactionRows;
    }

    @Override
    public String toString() {
        return String.format("Anomaly[customer=%s, type=%s, score=%d, level=%s]",
                customer, type.getLabel(), riskScore, riskLevel);
    }
}
