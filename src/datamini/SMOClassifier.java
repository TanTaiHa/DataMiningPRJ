package datamini;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

import java.util.Random;

public class SMOClassifier {
    public static double run(Instances data) throws Exception {
        SMO smo = new SMO();
        smo.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(smo, data, 10, new Random(1));

        System.out.println("=== SMO (SVM) ===");
        System.out.println(eval.toSummaryString("\n=== Evaluation Results ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        return eval.pctCorrect();
    }
}
