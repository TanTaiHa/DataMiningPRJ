package datamini;

import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.Instance;

import java.io.FileWriter;
import java.io.PrintWriter;

public class EMClusterer {
    public static void run(Instances data) throws Exception {
        EM em = new EM();
        em.buildClusterer(data);

        System.out.println("=== EM Clustering Result ===");
        System.out.println(em);

        PrintWriter writer = new PrintWriter(new FileWriter("em-cluster-output.txt"));
        writer.println("=== EM Clustering Result ===");
        writer.println(em);

        writer.println("\n=== Cluster Assignments for Instances ===");
        System.out.println("\n=== Cluster Assignments for Instances ===");

        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            int cluster = em.clusterInstance(inst);
            writer.printf("Instance %d => Cluster %d\n", i + 1, cluster);
            System.out.printf("Instance %d => Cluster %d\n", i + 1, cluster);
        }

        writer.close();
        System.out.println("\nCluster assignments saved to em-cluster-output.txt ✅");
    }
}
