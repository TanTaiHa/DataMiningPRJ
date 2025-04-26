package datamini;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import java.util.Random;

public class LogisticClassifier {
    public static double run(Instances data) throws Exception {
        Logistic logistic = new Logistic();
        logistic.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(logistic, data, 10, new Random(1));

        System.out.println("=== Logistic Regression ===");
        System.out.println(eval.toSummaryString("\n=== Evaluation Results ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        return eval.pctCorrect();
    }
}
