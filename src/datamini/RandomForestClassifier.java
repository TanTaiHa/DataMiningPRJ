package datamini;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.util.Random;

public class RandomForestClassifier {
    public static double run(Instances data) throws Exception {
        RandomForest rf = new RandomForest();
        rf.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(rf, data, 10, new Random(1));
        
        // In đầy đủ các phần kết quả
        System.out.println("=== Random Forest ===");
        System.out.println(eval.toSummaryString("\n=== Evaluation Results ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        return eval.pctCorrect(); // trả về Accuracy
    }
}
