package com.anomaly.io;

import com.anomaly.model.Transaction;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Parses the financial transaction CSV file into a list of {@link Transaction} objects.
 */
public class TransactionParser {

    /**
     * Reads a CSV file and returns all transactions.
     *
     * @param filePath path to the CSV file
     * @return list of parsed transactions
     * @throws IOException if the file cannot be read
     */
    public List<Transaction> parse(String filePath) throws IOException {
        List<Transaction> transactions = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String header = reader.readLine(); // skip header
            if (header == null) {
                return transactions;
            }

            String line;
            int rowIndex = 1;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                Transaction txn = parseLine(line, rowIndex);
                if (txn != null) {
                    transactions.add(txn);
                }
                rowIndex++;
            }
        }

        return transactions;
    }

    private Transaction parseLine(String line, int rowIndex) {
        String[] parts = line.split(",");
        if (parts.length < 11) {
            System.err.println("Skipping malformed row " + rowIndex + ": " + line);
            return null;
        }

        try {
            int step = Integer.parseInt(parts[0].trim());
            String type = parts[1].trim();
            double amount = Double.parseDouble(parts[2].trim());
            String nameOrig = parts[3].trim();
            double oldBalanceOrg = Double.parseDouble(parts[4].trim());
            double newBalanceOrig = Double.parseDouble(parts[5].trim());
            String nameDest = parts[6].trim();
            double oldBalanceDest = Double.parseDouble(parts[7].trim());
            double newBalanceDest = Double.parseDouble(parts[8].trim());
            int isFraud = Integer.parseInt(parts[9].trim());
            int isFlaggedFraud = Integer.parseInt(parts[10].trim());

            return new Transaction(step, type, amount, nameOrig, oldBalanceOrg,
                    newBalanceOrig, nameDest, oldBalanceDest, newBalanceDest,
                    isFraud, isFlaggedFraud, rowIndex);
        } catch (NumberFormatException e) {
            System.err.println("Skipping row " + rowIndex + " due to parse error: " + e.getMessage());
            return null;
        }
    }
}
