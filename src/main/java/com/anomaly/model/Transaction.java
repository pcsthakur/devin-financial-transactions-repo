package com.anomaly.model;

/**
 * Represents a single financial transaction parsed from the CSV data.
 */
public class Transaction {

    private final int step;
    private final String type;
    private final double amount;
    private final String nameOrig;
    private final double oldBalanceOrg;
    private final double newBalanceOrig;
    private final String nameDest;
    private final double oldBalanceDest;
    private final double newBalanceDest;
    private final int isFraud;
    private final int isFlaggedFraud;
    private final int rowIndex;

    public Transaction(int step, String type, double amount, String nameOrig,
                       double oldBalanceOrg, double newBalanceOrig,
                       String nameDest, double oldBalanceDest, double newBalanceDest,
                       int isFraud, int isFlaggedFraud, int rowIndex) {
        this.step = step;
        this.type = type;
        this.amount = amount;
        this.nameOrig = nameOrig;
        this.oldBalanceOrg = oldBalanceOrg;
        this.newBalanceOrig = newBalanceOrig;
        this.nameDest = nameDest;
        this.oldBalanceDest = oldBalanceDest;
        this.newBalanceDest = newBalanceDest;
        this.isFraud = isFraud;
        this.isFlaggedFraud = isFlaggedFraud;
        this.rowIndex = rowIndex;
    }

    public int getStep() {
        return step;
    }

    public String getType() {
        return type;
    }

    public double getAmount() {
        return amount;
    }

    public String getNameOrig() {
        return nameOrig;
    }

    public double getOldBalanceOrg() {
        return oldBalanceOrg;
    }

    public double getNewBalanceOrig() {
        return newBalanceOrig;
    }

    public String getNameDest() {
        return nameDest;
    }

    public double getOldBalanceDest() {
        return oldBalanceDest;
    }

    public double getNewBalanceDest() {
        return newBalanceDest;
    }

    public int getIsFraud() {
        return isFraud;
    }

    public int getIsFlaggedFraud() {
        return isFlaggedFraud;
    }

    public int getRowIndex() {
        return rowIndex;
    }

    @Override
    public String toString() {
        return String.format("Transaction[row=%d, step=%d, type=%s, amount=%.2f, from=%s, to=%s]",
                rowIndex, step, type, amount, nameOrig, nameDest);
    }
}
