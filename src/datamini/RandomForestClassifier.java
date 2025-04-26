package datamini;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.util.Random;

public class RandomForestClassifier {
    public static void run(Instances data) throws Exception {
        RandomForest rf = new RandomForest();
        rf.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(rf, data, 10, new Random(1));
        System.out.println("=== Random Forest ===");
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }
}
