package rand_opt;


import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Scanner;
import java.util.Arrays;

public class phishing {
    private static Instance[] instances = initializeInstances();
    private static Instance[] train_set = Arrays.copyOfRange(instances, 0, 7738);
    private static Instance[] test_set = Arrays.copyOfRange(instances, 7738, 11055);

    private static DataSet dataset = new DataSet(train_set);

    private static int inputLayer = 30, hiddenLayer = 16, outputLayer = 1, trainIterations = 5000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] opt = new OptimizationAlgorithm[3];
    private static String[] optNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat decform = new DecimalFormat("0.000");

    public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
//            String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
//            Path p = Paths.get(full_path);
            if (final_result) {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwriter = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwriter) {
                    pwriter.println(results);
                    pwriter.close();
                }
            } else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        String final_result = "";

        for (int i = 0; i < opt.length; i++) {
            networks[i] = factory.createClassificationNetwork (
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(dataset, networks[i], measure);
        }

        opt[0] = new RandomizedHillClimbing(nnop[0]);
        opt[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        opt[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        int[] iterations = {20, 100, 500, 1000, 2500, 5000};
        for (int trainIterations : iterations) {
            results = "";
            for (int i = 0; i < opt.length; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(opt[i], networks[i], optNames[i], trainIterations);
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance bestLearner = opt[i].getOptimal();
                networks[i].setWeights(bestLearner.getData());

                // Statistics of Training Data
                double predicted, target;
                start = System.nanoTime();
                for (int j = 0; j < train_set.length; j++) {
                    networks[i].setInputValues(train_set[j].getData());
                    networks[i].run();

                    target = Double.parseDouble(train_set[j].getLabel().toString());
                    predicted = Double.parseDouble(networks[i].getOutputValues().toString());

                    double dummy = Math.abs(predicted - target) < 0.5 ? correct++ : incorrect++;
                }

                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTraining Results for " + optNames[i] +
                        ": \nCorrect classification " + correct + "\nMisclassification " + incorrect +
                        "\n Accuracy: " + decform.format(correct / (correct + incorrect) * 100) + "%" +
                        "\nTraining time: " + decform.format(trainingTime) + " seconds" +
                        "\nTesting time: " + decform.format(testingTime) + " seconds\n";

                final_result = optNames[i] + ", " + trainIterations +
                        ", training accuracy," + decform.format(correct / (correct + incorrect) * 100) + "%, " +
                        "training time," + decform.format(trainingTime) +
                        ", testing time," + decform.format(testingTime);
                write_output_to_file("Optimization_Results", "phishing_rhc_results.csv", final_result, true);

                // Statistics of Test set
                start = System.nanoTime();
                correct = 0;
                incorrect = 0;
                for (int j = 0; j < test_set.length; j++) {
                    networks[i].setInputValues(test_set[j].getData());
                    networks[i].run();

                    target = Double.parseDouble(test_set[j].getLabel().toString());
                    predicted = Double.parseDouble(networks[i].getOutputValues().toString());

                    double dummy = Math.abs(predicted - target) < 0.5 ? correct++ : incorrect++;
                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTest Results for " + optNames[i] +
                        ": \nCorrect classification " + correct + "\nMisclassification " + incorrect +
                        "\nAccuracy: " + decform.format(correct / (correct + incorrect) * 100) + "%" +
                        "\nTraining time: " + decform.format(trainingTime) + " seconds" +
                        "\nTesting time: " + decform.format(testingTime) + " seconds\n";

                final_result = optNames[i] + ", " + trainIterations +
                        ", testing accuracy," + decform.format(correct / (correct + incorrect) * 100) + "%" +
                        ", training time," + decform.format(trainingTime) + ", testing time," + decform.format(testingTime);
                write_output_to_file("Optimization_Results", "phishing_results.csv", final_result, true);
            }
            System.out.println("results for iterations: " + trainIterations + "--------------------");
            System.out.println(results);
        }
    }

    private static void train(OptimizationAlgorithm opt, BackPropagationNetwork network, String optName, int iterations) {
        int trainIterations = iterations;
        for (int i = 0; i < trainIterations; i++) {
            opt.train();

            double train_error = 0;
            for (int j = 0; j < train_set.length; j++) {
                network.setInputValues(train_set[j].getData());
                network.run();

                Instance target = train_set[j].getLabel(), score = new Instance(network.getOutputValues());
                score.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                train_error += measure.value(target, score);
            }
        }
    }


    private static Instance[] initializeInstances() {
        double[][][] attributes = new double[11055][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("../dataset/phishing.csv")));

            // for each sample
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[30];  // 30 attributes
                attributes[i][1] = new double[1];   // classification

                // read features
                for(int j = 0; j < 30; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        } catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];
        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0] < 0 ? 0 : 1));
        }

        return instances;
    }
}