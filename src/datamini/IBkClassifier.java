package datamini;



import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.util.Random;

public class IBkClassifier {
    public static double run(Instances data) throws Exception {
        IBk ibk = new IBk();
        ibk.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(ibk, data, 10, new Random(1));

        System.out.println("=== IBk (k-NN) ===");
        System.out.println(eval.toSummaryString("\n=== Evaluation Results ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        return eval.pctCorrect();
    }
}
