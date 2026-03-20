package com.anomaly.model;

/**
 * Enumeration of the anomaly patterns detected by the system.
 */
public enum AnomalyType {

    REPEATED_HIGH_VALUE("Repeated High-Value Transactions"),
    TRANSFER_THEN_CASH_OUT("Transfer Followed by Cash-Out"),
    SUDDEN_AMOUNT_INCREASE("Sudden Increase in Transaction Amount");

    private final String label;

    AnomalyType(String label) {
        this.label = label;
    }

    public String getLabel() {
        return label;
    }
}
