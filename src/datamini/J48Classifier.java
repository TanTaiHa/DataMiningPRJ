package datamini;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.util.Random;

public class J48Classifier {
    public static void run(Instances data) throws Exception {
        J48 tree = new J48();
        tree.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(tree, data, 10, new Random(1));
        System.out.println("=== J48 ===");
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }
}
