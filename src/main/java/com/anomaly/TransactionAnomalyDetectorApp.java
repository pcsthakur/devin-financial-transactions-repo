package com.anomaly;

import com.anomaly.detector.AnomalyDetector;
import com.anomaly.io.ReportGenerator;
import com.anomaly.io.TransactionParser;
import com.anomaly.model.Anomaly;
import com.anomaly.model.Transaction;

import java.io.IOException;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Entry point for the Transaction Anomaly Detection system.
 *
 * <p>Reads financial transactions from a CSV file, detects anomalous
 * sequences per customer, and writes a risk report.
 *
 * <p>Detected patterns:
 * <ol>
 *   <li>Repeated high-value transactions</li>
 *   <li>Transfer followed by cash-out</li>
 *   <li>Sudden increase in transaction amounts</li>
 * </ol>
 *
 * <p>Usage: {@code java com.anomaly.TransactionAnomalyDetectorApp [csv_path] [report_path]}
 */
public class TransactionAnomalyDetectorApp {

    private static final String DEFAULT_DATA_PATH = "data/Example1.csv";
    private static final String DEFAULT_REPORT_PATH = "reports/anomaly_detection_report.csv";

    public static void main(String[] args) {
        String dataPath = args.length > 0 ? args[0] : DEFAULT_DATA_PATH;
        String reportPath = args.length > 1 ? args[1] : DEFAULT_REPORT_PATH;

        System.out.println("==========================================================");
        System.out.println("  Transaction Anomaly Detection System");
        System.out.println("==========================================================");

        // 1. Parse transactions
        TransactionParser parser = new TransactionParser();
        List<Transaction> transactions;
        try {
            System.out.printf("Loading transactions from %s ...%n", dataPath);
            transactions = parser.parse(dataPath);
            System.out.printf("Loaded %d transactions.%n%n", transactions.size());
        } catch (IOException e) {
            System.err.println("Error reading data file: " + e.getMessage());
            return;
        }

        if (transactions.isEmpty()) {
            System.out.println("No transactions found. Exiting.");
            return;
        }

        // 2. Detect anomalies
        System.out.println("Running anomaly detection ...");
        AnomalyDetector detector = new AnomalyDetector();
        List<Anomaly> anomalies = detector.detectAll(transactions);
        System.out.printf("Detected %d anomalies.%n%n", anomalies.size());

        // 3. Print summary
        printSummary(anomalies);

        // 4. Generate report
        ReportGenerator reportGen = new ReportGenerator();
        try {
            reportGen.generate(anomalies, reportPath);
            System.out.printf("%nReport saved to %s%n", reportPath);
        } catch (IOException e) {
            System.err.println("Error writing report: " + e.getMessage());
        }

        System.out.println("\nDone.");
    }

    private static void printSummary(List<Anomaly> anomalies) {
        System.out.println("==========================================================");
        System.out.println("  ANOMALY DETECTION SUMMARY");
        System.out.println("==========================================================");

        long high = anomalies.stream().filter(a -> "HIGH".equals(a.getRiskLevel())).count();
        long medium = anomalies.stream().filter(a -> "MEDIUM".equals(a.getRiskLevel())).count();
        long low = anomalies.stream().filter(a -> "LOW".equals(a.getRiskLevel())).count();

        System.out.printf("Total anomalies detected : %d%n", anomalies.size());
        System.out.printf("  HIGH risk   : %d%n", high);
        System.out.printf("  MEDIUM risk : %d%n", medium);
        System.out.printf("  LOW risk    : %d%n", low);

        // Breakdown by type
        System.out.println("\nBreakdown by anomaly type:");
        Map<String, Long> byType = anomalies.stream()
                .collect(Collectors.groupingBy(a -> a.getType().getLabel(), Collectors.counting()));
        byType.forEach((type, count) ->
                System.out.printf("  %-45s : %d%n", type, count));

        // Top HIGH-risk anomalies
        if (high > 0) {
            System.out.println("\nTop HIGH-risk anomalies:");
            anomalies.stream()
                    .filter(a -> "HIGH".equals(a.getRiskLevel()))
                    .sorted(Comparator.comparingInt(Anomaly::getRiskScore).reversed())
                    .limit(10)
                    .forEach(a -> System.out.printf(
                            "  [Score %3d] %-15s | %-40s | %s%n",
                            a.getRiskScore(), a.getCustomer(),
                            a.getType().getLabel(),
                            truncate(a.getExplanation(), 80)));
        }

        System.out.println("==========================================================");
    }

    private static String truncate(String text, int maxLen) {
        if (text.length() <= maxLen) {
            return text;
        }
        return text.substring(0, maxLen - 3) + "...";
    }
}
