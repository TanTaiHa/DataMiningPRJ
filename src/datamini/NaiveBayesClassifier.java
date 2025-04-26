package datamini;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

public class NaiveBayesClassifier {
    public static void run(Instances data) throws Exception {
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(nb, data, 10, new Random(1));
        System.out.println("=== Naive Bayes ===");
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }
}
