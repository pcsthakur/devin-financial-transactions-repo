package com.anomaly.io;

import com.anomaly.model.Anomaly;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

/**
 * Writes detected anomalies to a CSV report file.
 */
public class ReportGenerator {

    private static final String HEADER =
            "customer,anomaly_type,risk_score,risk_level,transaction_rows,explanation";

    /**
     * Generates a CSV report of all detected anomalies.
     *
     * @param anomalies  the anomalies to report
     * @param outputPath path for the output CSV file
     * @throws IOException if the file cannot be written
     */
    public void generate(List<Anomaly> anomalies, String outputPath) throws IOException {
        File parent = new File(outputPath).getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            writer.write(HEADER);
            writer.newLine();

            for (Anomaly anomaly : anomalies) {
                writer.write(formatRow(anomaly));
                writer.newLine();
            }
        }
    }

    private String formatRow(Anomaly anomaly) {
        StringBuilder rows = new StringBuilder();
        for (int i = 0; i < anomaly.getTransactionRows().size(); i++) {
            if (i > 0) {
                rows.append(";");
            }
            rows.append(anomaly.getTransactionRows().get(i));
        }

        return String.format("%s,%s,%d,%s,%s,\"%s\"",
                anomaly.getCustomer(),
                anomaly.getType().getLabel(),
                anomaly.getRiskScore(),
                anomaly.getRiskLevel(),
                rows.toString(),
                anomaly.getExplanation().replace("\"", "'"));
    }
}
