package datamini;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("datamini/segment-challenge.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        J48Classifier.run(data);
        NaiveBayesClassifier.run(data);
        RandomForestClassifier.run(data);
        // Add more classifiers here
    }
}
