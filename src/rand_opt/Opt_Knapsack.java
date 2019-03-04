package rand_opt;


import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.*;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;

/**
 * Adapted from Knapsack test example
 * by Andrew Guillory gtg008g@mail.gatech.edu  @version 1.0
 */

public class Opt_Knapsack {
    private static final Random random = new Random();
    private static final int NUM_TIEMS = 40;
    private static final int COPIES_EACH = 4;
    private static final double ITEM_MAX_WEIGHT = 50;
    private static final double ITEM_MAX_VOLUME = 50;
    private static final double KNAPSACK_VOLUME =
            ITEM_MAX_VOLUME * NUM_TIEMS * COPIES_EACH * .4;

    public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) +  "/" + file_name;
            Path p = Paths.get(full_path);
            if (final_result) {
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwriter = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized(pwriter) {
                    pwriter.println(results);
                    pwriter.close();
                }
            } else {
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        double start, end, elapse;
        int[] iterations = {20, 100, 500, 1000, 2500, 5000, 10000};
        int testRuns = 10;

        for (int iter: iterations) {
            int sum_rhc = 0, sum_sa = 0, sum_ga = 0, sum_mimic = 0;
            double elapse_rhc = 0, elapse_sa = 0, elapse_ga = 0, elapse_mimic = 0;

            for (int j = 0; j < testRuns; j++) {

                // create items for knapsack
                int[] copies = new int[NUM_TIEMS];
                Arrays.fill(copies, COPIES_EACH);
                double[] weights = new double[NUM_TIEMS];
                double[] volumes = new double[NUM_TIEMS];
                for (int i = 0; i < NUM_TIEMS; i++) {
                    weights[i] = random.nextDouble() * ITEM_MAX_WEIGHT;
                    volumes[i] = random.nextDouble() * ITEM_MAX_VOLUME;
                }
                int[] ranges = new int[NUM_TIEMS];
                Arrays.fill(ranges, COPIES_EACH + 1);

                EvaluationFunction evalfunc = new KnapsackEvaluationFunction(weights, volumes,
                                                KNAPSACK_VOLUME, copies);
                Distribution unidist = new DiscreteUniformDistribution(ranges);
                NeighborFunction neighfunc = new DiscreteChangeOneNeighbor(ranges);
                MutationFunction mutfunc = new DiscreteChangeOneMutation(ranges);
                CrossoverFunction crofunc = new UniformCrossOver();
                Distribution deptree = new DiscreteDependencyTree(.1, ranges);
                HillClimbingProblem hcp = new GenericHillClimbingProblem(evalfunc, unidist, neighfunc);
                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(evalfunc, unidist, mutfunc, crofunc);
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(evalfunc, unidist, deptree);

                start = System.nanoTime();
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
                fit.train();
                end = System.nanoTime();
                elapse = end - start;
                elapse /= Math.pow(10, 9);
                sum_rhc += evalfunc.value(rhc.getOptimal());
                elapse_rhc += elapse;
                System.out.println("rhc: " + evalfunc.value(rhc.getOptimal()));
                System.out.println(elapse);

                start = System.nanoTime();
                SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
                fit = new FixedIterationTrainer(sa, iter);
                fit.train();
                end = System.nanoTime();
                elapse = end - start;
                elapse /= Math.pow(10, 9);
                sum_sa += evalfunc.value(sa.getOptimal());
                elapse_sa += elapse;
                System.out.println("sa: " + evalfunc.value(sa.getOptimal()));
                System.out.println(elapse);

                start = System.nanoTime();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
                fit = new FixedIterationTrainer(ga, iter);
                fit.train();
                end = System.nanoTime();
                elapse = end - start;
                elapse /= Math.pow(10, 9);
                sum_ga += evalfunc.value(ga.getOptimal());
                elapse_ga += elapse;
                System.out.println("ga: " + evalfunc.value(ga.getOptimal()));
                System.out.println(elapse);

                start = System.nanoTime();
                MIMIC mimic = new MIMIC(200, 20, pop);
                fit = new FixedIterationTrainer(mimic, iter);
                fit.train();
                end = System.nanoTime();
                elapse = end - start;
                elapse /= Math.pow(10, 9);
                sum_mimic += evalfunc.value(mimic.getOptimal());
                elapse_mimic += elapse;
                System.out.println("Mimic: " + evalfunc.value(mimic.getOptimal()));
                System.out.println(elapse);
            }

            int average_rhc = sum_rhc / testRuns;
            int average_sa = sum_sa / testRuns;
            int average_ga = sum_ga / testRuns;
            int average_mimic = sum_mimic / testRuns;

            double averageelapse_rhc = elapse_rhc / testRuns;
            double averageelapse_sa = elapse_sa / testRuns;
            double averageelapse_ga = elapse_ga / testRuns;
            double averageelapse_mimic = elapse_mimic / testRuns;

            System.out.println("-------------------");
            System.out.println("Iteration " + iter);
            System.out.println("rhc average, " + average_rhc + ", elapse average, " + averageelapse_rhc);
            System.out.println("sa average, " + average_sa + ", elapse average, " + averageelapse_sa);
            System.out.println("ga average, " + average_ga + ", elapse average, " + averageelapse_ga);
            System.out.println("mimic average, " + average_mimic + ", elapse average, " + averageelapse_mimic);

            String final_result = "";
            final_result =
                    "rhc" + ", " + iter + ", " + Integer.toString(average_rhc) + ", " + Double.toString(averageelapse_rhc) + ", " +
                            "sa" + ", " + iter + ", " + Integer.toString(average_sa) + ", " + Double.toString(averageelapse_sa) + ", " +
                            "ga" + ", " + iter + ", " + Integer.toString(average_ga) + ", " + Double.toString(averageelapse_ga) + ", " +
                            "mimic" + ", " + iter + ", " + Integer.toString(average_mimic) + ", " + Double.toString(averageelapse_mimic);

            write_output_to_file("Optimization_Results", "knapsack_results.csv", final_result, true);
        }

    }
}
