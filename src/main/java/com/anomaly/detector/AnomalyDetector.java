package com.anomaly.detector;

import com.anomaly.model.Anomaly;
import com.anomaly.model.AnomalyType;
import com.anomaly.model.Transaction;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Detects anomalous transaction sequences for each customer.
 *
 * <p>Three patterns are detected:
 * <ol>
 *   <li><b>Repeated high-value transactions</b> &ndash; multiple high-value
 *       transactions involving the same account (as originator or as a
 *       destination that collects funds from many sources).</li>
 *   <li><b>Transfer followed by cash-out</b> &ndash; a TRANSFER whose
 *       destination account subsequently originates a CASH_OUT, a TRANSFER
 *       paired with a CASH_OUT of matching amount in the same time step
 *       (layering pattern), or a customer performing both types.</li>
 *   <li><b>Sudden increase in transaction amount</b> &ndash; a transaction
 *       whose amount is a statistical outlier compared to transactions of the
 *       same type, or whose amount far exceeds the customer's account
 *       balance.</li>
 * </ol>
 */
public class AnomalyDetector {

    /** Transactions above this amount are considered high-value. */
    private static final double HIGH_AMOUNT_THRESHOLD = 10_000.0;

    /** Minimum incoming high-value transactions to flag a destination as a collection point. */
    private static final int COLLECTION_POINT_MIN_COUNT = 2;

    /**
     * A transaction amount must exceed the type-average by this factor to
     * be flagged as a sudden increase / outlier.
     */
    private static final double SUDDEN_INCREASE_FACTOR = 3.0;

    /**
     * Maximum step gap between a TRANSFER and a subsequent CASH_OUT to be
     * considered part of the same suspicious sequence.
     */
    private static final int TRANSFER_CASHOUT_MAX_STEP_GAP = 1;

    /**
     * Tolerance for matching amounts between TRANSFER and CASH_OUT (as a
     * fraction of the larger amount).
     */
    private static final double AMOUNT_MATCH_TOLERANCE = 0.01;

    // -----------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------

    /**
     * Run all anomaly detectors against the provided transactions and return
     * every anomaly found.
     */
    public List<Anomaly> detectAll(List<Transaction> transactions) {
        List<Anomaly> anomalies = new ArrayList<>();

        anomalies.addAll(detectRepeatedHighValue(transactions));
        anomalies.addAll(detectTransferThenCashOut(transactions));
        anomalies.addAll(detectSuddenAmountIncrease(transactions));

        return anomalies;
    }

    // -----------------------------------------------------------------
    // Pattern 1 – Repeated high-value transactions
    // -----------------------------------------------------------------

    /**
     * Detects two sub-patterns:
     * <ul>
     *   <li>A single customer making multiple high-value transactions.</li>
     *   <li>A destination account receiving multiple high-value incoming
     *       transactions from different originators (collection point).</li>
     * </ul>
     */
    private List<Anomaly> detectRepeatedHighValue(List<Transaction> transactions) {
        List<Anomaly> results = new ArrayList<>();

        // Sub-pattern A: same originator, multiple high-value transactions
        Map<String, List<Transaction>> byOrig = groupBy(transactions, Transaction::getNameOrig);
        for (Map.Entry<String, List<Transaction>> entry : byOrig.entrySet()) {
            List<Transaction> highValue = entry.getValue().stream()
                    .filter(t -> t.getAmount() > HIGH_AMOUNT_THRESHOLD)
                    .collect(Collectors.toList());
            if (highValue.size() >= COLLECTION_POINT_MIN_COUNT) {
                results.add(buildRepeatedHighValueAnomaly(entry.getKey(), highValue,
                        "originator"));
            }
        }

        // Sub-pattern B: same destination receives multiple high-value transactions
        Map<String, List<Transaction>> byDest = groupBy(transactions, Transaction::getNameDest);
        for (Map.Entry<String, List<Transaction>> entry : byDest.entrySet()) {
            List<Transaction> highValue = entry.getValue().stream()
                    .filter(t -> t.getAmount() > HIGH_AMOUNT_THRESHOLD)
                    .collect(Collectors.toList());
            if (highValue.size() >= COLLECTION_POINT_MIN_COUNT) {
                Set<String> uniqueOriginators = highValue.stream()
                        .map(Transaction::getNameOrig)
                        .collect(Collectors.toSet());
                if (uniqueOriginators.size() >= COLLECTION_POINT_MIN_COUNT) {
                    results.add(buildCollectionPointAnomaly(entry.getKey(), highValue,
                            uniqueOriginators));
                }
            }
        }

        return results;
    }

    private Anomaly buildRepeatedHighValueAnomaly(String customer,
                                                   List<Transaction> highValue,
                                                   String role) {
        List<Integer> rows = highValue.stream()
                .map(Transaction::getRowIndex).collect(Collectors.toList());

        int base = 40;
        int countBonus = Math.min((highValue.size() - COLLECTION_POINT_MIN_COUNT) * 10, 30);
        boolean hasRiskyType = highValue.stream()
                .anyMatch(t -> "CASH_OUT".equals(t.getType()) || "TRANSFER".equals(t.getType()));
        int typeBonus = hasRiskyType ? 15 : 0;
        double maxAmount = highValue.stream().mapToDouble(Transaction::getAmount).max().orElse(0);
        int amountBonus = maxAmount > 500_000 ? 15 : maxAmount > 100_000 ? 10 : 0;
        int score = Math.min(base + countBonus + typeBonus + amountBonus, 100);

        double totalAmount = highValue.stream().mapToDouble(Transaction::getAmount).sum();
        String explanation = String.format(
                "Account %s (%s) has %d high-value transactions (each > %.0f) "
                        + "totalling %.2f. Types: %s",
                customer, role, highValue.size(), HIGH_AMOUNT_THRESHOLD, totalAmount,
                highValue.stream().map(Transaction::getType).collect(Collectors.joining(", ")));

        return new Anomaly(customer, AnomalyType.REPEATED_HIGH_VALUE, score, explanation, rows);
    }

    private Anomaly buildCollectionPointAnomaly(String destAccount,
                                                 List<Transaction> incoming,
                                                 Set<String> originators) {
        List<Integer> rows = incoming.stream()
                .map(Transaction::getRowIndex).collect(Collectors.toList());

        int base = 50;
        int countBonus = Math.min((incoming.size() - COLLECTION_POINT_MIN_COUNT) * 8, 25);
        boolean allTransfer = incoming.stream().allMatch(t -> "TRANSFER".equals(t.getType()));
        int typeBonus = allTransfer ? 15 : 10;
        double totalAmount = incoming.stream().mapToDouble(Transaction::getAmount).sum();
        int amountBonus = totalAmount > 1_000_000 ? 15 : totalAmount > 500_000 ? 10 : 0;
        int score = Math.min(base + countBonus + typeBonus + amountBonus, 100);

        String explanation = String.format(
                "Destination account %s received %d high-value transactions from %d "
                        + "different originators (%s) totalling %.2f — possible collection point",
                destAccount, incoming.size(), originators.size(),
                String.join(", ", originators), totalAmount);

        String firstOriginator = incoming.get(0).getNameOrig();
        return new Anomaly(firstOriginator, AnomalyType.REPEATED_HIGH_VALUE,
                score, explanation, rows);
    }

    // -----------------------------------------------------------------
    // Pattern 2 – Transfer followed by cash-out
    // -----------------------------------------------------------------

    /**
     * Detects suspicious TRANSFER → CASH_OUT sequences:
     * <ul>
     *   <li><b>Direct chain:</b> A TRANSFER to account X, followed by a
     *       CASH_OUT originating from account X within a short time window.</li>
     *   <li><b>Amount-matched layering:</b> A TRANSFER and CASH_OUT of the
     *       same (or very similar) amount in the same time step, indicating
     *       possible layering through mule accounts.</li>
     *   <li><b>Same-customer pair:</b> A customer who performs both a TRANSFER
     *       and a CASH_OUT.</li>
     * </ul>
     */
    private List<Anomaly> detectTransferThenCashOut(List<Transaction> transactions) {
        List<Anomaly> results = new ArrayList<>();
        Set<String> reported = new HashSet<>();

        List<Transaction> transfers = transactions.stream()
                .filter(t -> "TRANSFER".equals(t.getType()))
                .sorted(Comparator.comparingInt(Transaction::getStep))
                .collect(Collectors.toList());

        List<Transaction> cashOuts = transactions.stream()
                .filter(t -> "CASH_OUT".equals(t.getType()))
                .sorted(Comparator.comparingInt(Transaction::getStep))
                .collect(Collectors.toList());

        // Index CASH_OUTs by originator for direct-chain lookup
        Map<String, List<Transaction>> cashOutByOrig =
                groupBy(cashOuts, Transaction::getNameOrig);

        for (Transaction transfer : transfers) {
            // Case A: TRANSFER destination later originates a CASH_OUT
            List<Transaction> destCashOuts =
                    cashOutByOrig.getOrDefault(transfer.getNameDest(), List.of());
            for (Transaction co : destCashOuts) {
                if (co.getStep() >= transfer.getStep()
                        && co.getStep() <= transfer.getStep() + TRANSFER_CASHOUT_MAX_STEP_GAP) {
                    String key = "A|" + transfer.getRowIndex() + "|" + co.getRowIndex();
                    if (reported.add(key)) {
                        results.add(buildTransferCashOutAnomaly(transfer, co,
                                String.format("TRANSFER to %s followed by CASH_OUT from that account",
                                        transfer.getNameDest())));
                    }
                }
            }

            // Case B: TRANSFER and CASH_OUT with matching amounts in same step window
            for (Transaction co : cashOuts) {
                if (co.getStep() >= transfer.getStep()
                        && co.getStep() <= transfer.getStep() + TRANSFER_CASHOUT_MAX_STEP_GAP
                        && amountsMatch(transfer.getAmount(), co.getAmount())) {
                    String key = "B|" + transfer.getRowIndex() + "|" + co.getRowIndex();
                    if (reported.add(key)) {
                        results.add(buildTransferCashOutAnomaly(transfer, co,
                                String.format(
                                        "TRANSFER and CASH_OUT with matching amount %.2f in step %d "
                                                + "(possible layering through mule accounts)",
                                        transfer.getAmount(), transfer.getStep())));
                    }
                }
            }
        }

        // Case C: same customer originates both a TRANSFER and CASH_OUT
        Map<String, List<Transaction>> byOrig = groupBy(transactions, Transaction::getNameOrig);
        for (Map.Entry<String, List<Transaction>> entry : byOrig.entrySet()) {
            List<Transaction> custTransfers = entry.getValue().stream()
                    .filter(t -> "TRANSFER".equals(t.getType())).collect(Collectors.toList());
            List<Transaction> custCashOuts = entry.getValue().stream()
                    .filter(t -> "CASH_OUT".equals(t.getType())).collect(Collectors.toList());
            if (!custTransfers.isEmpty() && !custCashOuts.isEmpty()) {
                for (Transaction tr : custTransfers) {
                    for (Transaction co : custCashOuts) {
                        String key = "C|" + tr.getRowIndex() + "|" + co.getRowIndex();
                        if (reported.add(key)) {
                            results.add(buildTransferCashOutAnomaly(tr, co,
                                    String.format("Customer %s performed both TRANSFER and CASH_OUT",
                                            entry.getKey())));
                        }
                    }
                }
            }
        }

        return results;
    }

    private boolean amountsMatch(double a, double b) {
        if (a == 0 && b == 0) {
            return false;
        }
        double max = Math.max(a, b);
        return Math.abs(a - b) / max <= AMOUNT_MATCH_TOLERANCE;
    }

    private Anomaly buildTransferCashOutAnomaly(Transaction transfer,
                                                 Transaction cashOut,
                                                 String pattern) {
        int score = 55;
        if (transfer.getAmount() > HIGH_AMOUNT_THRESHOLD) {
            score += 15;
        }
        if (cashOut.getAmount() > HIGH_AMOUNT_THRESHOLD) {
            score += 10;
        }
        if (transfer.getNewBalanceOrig() == 0) {
            score += 10;
        }
        if (amountsMatch(transfer.getAmount(), cashOut.getAmount())) {
            score += 10;
        }

        String explanation = String.format(
                "%s. TRANSFER: %s->%s amount=%.2f (step %d, row %d) | "
                        + "CASH_OUT: %s->%s amount=%.2f (step %d, row %d)",
                pattern,
                transfer.getNameOrig(), transfer.getNameDest(),
                transfer.getAmount(), transfer.getStep(), transfer.getRowIndex(),
                cashOut.getNameOrig(), cashOut.getNameDest(),
                cashOut.getAmount(), cashOut.getStep(), cashOut.getRowIndex());

        List<Integer> rows = List.of(transfer.getRowIndex(), cashOut.getRowIndex());
        String customer = transfer.getNameOrig();
        return new Anomaly(customer, AnomalyType.TRANSFER_THEN_CASH_OUT,
                Math.min(score, 100), explanation, rows);
    }

    // -----------------------------------------------------------------
    // Pattern 3 – Sudden increase in transaction amount
    // -----------------------------------------------------------------

    /**
     * Detects transactions whose amount is a significant outlier compared to
     * other transactions of the same type, or where the amount far exceeds
     * the originating account balance.
     */
    private List<Anomaly> detectSuddenAmountIncrease(List<Transaction> transactions) {
        List<Anomaly> results = new ArrayList<>();

        // Compute per-type statistics
        Map<String, List<Double>> amountsByType = new HashMap<>();
        for (Transaction txn : transactions) {
            amountsByType.computeIfAbsent(txn.getType(), k -> new ArrayList<>())
                    .add(txn.getAmount());
        }
        Map<String, Double> avgByType = new HashMap<>();
        for (Map.Entry<String, List<Double>> entry : amountsByType.entrySet()) {
            double avg = entry.getValue().stream()
                    .mapToDouble(Double::doubleValue).average().orElse(0);
            avgByType.put(entry.getKey(), avg);
        }

        for (Transaction txn : transactions) {
            List<String> reasons = new ArrayList<>();

            // Sub-check A: amount is an outlier relative to same-type average
            double typeAvg = avgByType.getOrDefault(txn.getType(), 0.0);
            double ratio = typeAvg > 0 ? txn.getAmount() / typeAvg : 0;
            boolean isOutlier = typeAvg > 0 && ratio > SUDDEN_INCREASE_FACTOR;
            if (isOutlier) {
                reasons.add(String.format(
                        "Amount %.2f is %.1fx the average %.2f for %s transactions",
                        txn.getAmount(), ratio, typeAvg, txn.getType()));
            }

            // Sub-check B: amount greatly exceeds originator balance
            boolean exceedsBalance = txn.getOldBalanceOrg() > 0
                    && txn.getAmount() > txn.getOldBalanceOrg() * SUDDEN_INCREASE_FACTOR;
            if (exceedsBalance) {
                reasons.add(String.format(
                        "Amount %.2f exceeds origin balance %.2f by %.1fx",
                        txn.getAmount(), txn.getOldBalanceOrg(),
                        txn.getAmount() / txn.getOldBalanceOrg()));
            }

            if (!reasons.isEmpty()) {
                int score = computeSuddenIncreaseScore(txn, typeAvg, isOutlier, exceedsBalance);
                String explanation = String.format(
                        "Customer %s (step %d, type %s): %s",
                        txn.getNameOrig(), txn.getStep(), txn.getType(),
                        String.join("; ", reasons));

                results.add(new Anomaly(txn.getNameOrig(),
                        AnomalyType.SUDDEN_AMOUNT_INCREASE,
                        score, explanation, List.of(txn.getRowIndex())));
            }
        }

        return results;
    }

    private int computeSuddenIncreaseScore(Transaction txn, double typeAvg,
                                            boolean isOutlier, boolean exceedsBalance) {
        int score = 35;
        if (isOutlier) {
            double ratio = txn.getAmount() / typeAvg;
            score += (int) Math.min((ratio - SUDDEN_INCREASE_FACTOR) * 5, 25);
        }
        if (exceedsBalance) {
            score += 15;
        }
        if ("CASH_OUT".equals(txn.getType()) || "TRANSFER".equals(txn.getType())) {
            score += 15;
        }
        if (txn.getAmount() > HIGH_AMOUNT_THRESHOLD) {
            score += 10;
        }
        return Math.min(score, 100);
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private <K> Map<K, List<Transaction>> groupBy(List<Transaction> txns,
                                                    Function<Transaction, K> keyFn) {
        Map<K, List<Transaction>> map = new HashMap<>();
        for (Transaction txn : txns) {
            map.computeIfAbsent(keyFn.apply(txn), k -> new ArrayList<>()).add(txn);
        }
        return map;
    }
}
