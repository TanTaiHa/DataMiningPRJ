package datamini;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("datamini/segment-challenge.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // Map lưu tên thuật toán và accuracy
        Map<String, Double> accuracies = new HashMap<>();

        accuracies.put("J48", J48Classifier.run(data));
        accuracies.put("RandomForest", RandomForestClassifier.run(data));
        accuracies.put("NaiveBayes", NaiveBayesClassifier.run(data));
        accuracies.put("SMO", SMOClassifier.run(data));
        accuracies.put("IBk", IBkClassifier.run(data));
        accuracies.put("Logistic", LogisticClassifier.run(data));

        // Tìm accuracy cao nhất
        double maxAccuracy = accuracies.values().stream().mapToDouble(v -> v).max().orElse(0);

        System.out.println("\n=== Algorithms with Highest Accuracy (" + maxAccuracy + "%) ===");
        for (String algo : accuracies.keySet()) {
            if (accuracies.get(algo) == maxAccuracy) {
                System.out.println("- " + algo);
            }
        }
    }
}
