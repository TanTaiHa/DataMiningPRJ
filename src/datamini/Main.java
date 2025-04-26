package datamini;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws Exception {
        // Load dữ liệu
        DataSource source = new DataSource("datamini/segment-challenge.arff");
        Instances data = source.getDataSet();

        // Gán cột class nếu chưa có (cho classification)
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

        // In tất cả độ chính xác
        System.out.println("\n=== Accuracy of All Algorithms ===");
        for (Map.Entry<String, Double> entry : accuracies.entrySet()) {
            System.out.printf("%-15s : %.2f%%\n", entry.getKey(), entry.getValue());
        }

        // Tìm thuật toán có độ chính xác cao nhất
        double maxAccuracy = accuracies.values().stream().mapToDouble(v -> v).max().orElse(0);

        System.out.println("\n=== Algorithms with Highest Accuracy (" + maxAccuracy + "%) ===");
        for (String algo : accuracies.keySet()) {
            if (accuracies.get(algo) == maxAccuracy) {
                System.out.println("- " + algo);
            }
        }

        // Chuyển sang clustering: bỏ class index
        data.setClassIndex(-1);

        // Gọi EM clustering
        System.out.println("\n=== Running EM Clustering ===");
        EMClusterer.run(data);
    }
}
