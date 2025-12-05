package se.kth.jabeja;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.log4j.Logger;

import se.kth.jabeja.config.Config;
import se.kth.jabeja.io.FileIO;
import se.kth.jabeja.rand.RandNoGenerator;

public class Jabeja {
  final static Logger logger = Logger.getLogger(Jabeja.class);
  private final Config config;
  private final HashMap<Integer, Node> entireGraph;
  private final List<Integer> nodeIds;
  private int numberOfSwaps;
  private int round;
  private float T;
  private boolean resultFileCreated = false;

  // -------------------------------------------------------------------
  public Jabeja(HashMap<Integer, Node> graph, Config config) {
    this.entireGraph = graph;
    this.nodeIds = new ArrayList<>(entireGraph.keySet());
    this.round = 0;
    this.numberOfSwaps = 0;
    this.config = config;
    this.T = config.getTemperature();
  }

  // -------------------------------------------------------------------
  public void startJabeja() throws IOException {
    for (round = 0; round < config.getRounds(); round++) {
      for (int id : entireGraph.keySet()) {
        sampleAndSwap(id);
      }

      // Apply simulated annealing
      saCoolDown();

      report();
    }
  }

  /**
   * Task 2: Geometric simulated annealing + restart
   */
  private void saCoolDown() {
    T = T * (1 - config.getDelta());

    if (T < 0.01)
      T = config.getTemperature();
  }

  private void sampleAndSwap(int nodeId) {
    Node nodep = entireGraph.get(nodeId);

    Node partner = null;

    // 1) LOCAL SEARCH FIRST (mandatory for Hybrid)
    Integer[] local = getNeighbors(nodep);
    partner = findPartner(nodeId, local);

    // 2) If LOCAL fails → GLOBAL search
    if (partner == null) {
      Integer[] randomSample = getSample(nodeId);
      partner = findPartner(nodeId, randomSample);
    }

    // 3) Perform the swap
    if (partner != null) {
      int pColor = nodep.getColor();
      int qColor = partner.getColor();

      nodep.setColor(qColor);
      partner.setColor(pColor);

      numberOfSwaps++;
    }
  }

  /**
   * Partner selection come in Algoritmo 1 del paper:
   * new = d_p(qColor)^alpha + d_q(pColor)^alpha
   * old = d_p(pColor)^alpha + d_q(qColor)^alpha
   * accetto se (new * T > old) e new > bestUtility
   * (T è la temperatura del simulated annealing).
   */
  public Node findPartner(int nodeId, Integer[] candidates) {

    Node nodep = entireGraph.get(nodeId);
    Node bestPartner = null;

    double bestUtility = 0;
    double alpha = 2.0; // Fixed as per paper

    for (int qId : candidates) {

      Node nodeq = entireGraph.get(qId);

      // Compute old degrees
      int d_pp = getDegree(nodep, nodep.getColor());
      int d_qq = getDegree(nodeq, nodeq.getColor());

      // Degrees after hypothetical swap
      int d_pq = getDegree(nodep, nodeq.getColor());
      int d_qp = getDegree(nodeq, nodep.getColor());

      double oldUtility = Math.pow(d_pp, alpha) + Math.pow(d_qq, alpha);

      double newUtility = Math.pow(d_pq, alpha) + Math.pow(d_qp, alpha);

      // -------- ACCEPTANCE RULE FROM PAPER --------
      boolean accept = newUtility / Math.pow(oldUtility, 1.0 / T) > 1.0;

      if (accept && newUtility > bestUtility) {
        bestUtility = newUtility;
        bestPartner = nodeq;
      }
    }
    return bestPartner;
  }

  /**
   * Count neighbors of a given color
   */
  private int getDegree(Node node, int colorId) {
    int degree = 0;
    for (int neighborId : node.getNeighbours()) {
      Node neighbor = entireGraph.get(neighborId);
      if (neighbor.getColor() == colorId) {
        degree++;
      }
    }
    return degree;
  }

  /**
   * Uniform random sample of the graph (global random view),
   * come richiesto dal paper per la policy R.
   */
  private Integer[] getSample(int currentNodeId) {
    int count = config.getUniformRandomSampleSize();
    int size = entireGraph.size();
    ArrayList<Integer> rndIds = new ArrayList<>();

    while (count > 0) {
      int rndId = nodeIds.get(RandNoGenerator.nextInt(size));
      if (rndId != currentNodeId && !rndIds.contains(rndId)) {
        rndIds.add(rndId);
        count--;
      }
    }

    return rndIds.toArray(new Integer[0]);
  }

  /**
   * Biased Random Sampling:
   * Prefer candidates that share the same color as p.
   * This helps reaching better partitions faster.
   */
  private Integer[] getBiasedSample(int currentNodeId, int color) {

    List<Integer> sameColor = new ArrayList<>();
    List<Integer> diffColor = new ArrayList<>();

    for (int id : nodeIds) {
      if (id == currentNodeId)
        continue;
      Node n = entireGraph.get(id);

      if (n.getColor() == color)
        sameColor.add(id);
      else
        diffColor.add(id);
    }

    // Shuffle both lists
    java.util.Collections.shuffle(sameColor);
    java.util.Collections.shuffle(diffColor);

    int sampleSize = config.getUniformRandomSampleSize();
    List<Integer> out = new ArrayList<>();

    // Take 70% from same-color, 30% from others
    int fromSame = Math.min((int) (sampleSize * 0.7), sameColor.size());
    int fromDiff = sampleSize - fromSame;

    out.addAll(sameColor.subList(0, fromSame));
    if (fromDiff > 0 && diffColor.size() > 0)
      out.addAll(diffColor.subList(0, Math.min(fromDiff, diffColor.size())));

    return out.toArray(new Integer[0]);
  }

  /**
   * Random subset of neighbors
   */
  private Integer[] getNeighbors(Node node) {
    ArrayList<Integer> list = node.getNeighbours();
    int count = config.getRandomNeighborSampleSize();
    int size = list.size();

    if (size <= count)
      return list.toArray(new Integer[0]);

    ArrayList<Integer> rndIds = new ArrayList<>();
    while (count > 0) {
      int index = RandNoGenerator.nextInt(size);
      int rndId = list.get(index);
      if (!rndIds.contains(rndId)) {
        rndIds.add(rndId);
        count--;
      }
    }

    return rndIds.toArray(new Integer[0]);
  }

  /**
   * Reporting metrics
   */
  private void report() throws IOException {
    int grayLinks = 0;
    int migrations = 0;

    for (int i : entireGraph.keySet()) {
      Node node = entireGraph.get(i);
      int nodeColor = node.getColor();

      if (nodeColor != node.getInitColor()) {
        migrations++;
      }

      ArrayList<Integer> neighbors = node.getNeighbours();
      if (neighbors != null) {
        for (int n : neighbors) {
          if (nodeColor != entireGraph.get(n).getColor())
            grayLinks++;
        }
      }
    }

    int edgeCut = grayLinks / 2;

    logger.info("round: " + round +
        ", edge cut:" + edgeCut +
        ", swaps: " + numberOfSwaps +
        ", migrations: " + migrations);

    saveToFile(edgeCut, migrations);
  }

  private void saveToFile(int edgeCuts, int migrations) throws IOException {
    String delimiter = "\t\t";
    File inputFile = new File(config.getGraphFilePath());

    String outputFilePath = config.getOutputDir() +
        File.separator +
        inputFile.getName() + "_" +
        "NS_" + config.getNodeSelectionPolicy() + "_" +
        "GICP_" + config.getGraphInitialColorPolicy() + "_" +
        "T_" + config.getTemperature() + "_" +
        "D_" + config.getDelta() + "_" +
        "RNSS_" + config.getRandomNeighborSampleSize() + "_" +
        "URSS_" + config.getUniformRandomSampleSize() + "_" +
        "A_" + config.getAlpha() + "_" +
        "R_" + config.getRounds() + ".txt";

    if (!resultFileCreated) {
      File outputDir = new File(config.getOutputDir());
      if (!outputDir.exists()) {
        if (!outputDir.mkdir()) {
          throw new IOException("Unable to create the output directory");
        }
      }
      String header = "# Migration is number of nodes that have changed color.";
      header += "\n\nRound" + delimiter + "Edge-Cut" + delimiter + "Swaps" + delimiter + "Migrations" + delimiter
          + "Skipped" + "\n";
      FileIO.write(header, outputFilePath);
      resultFileCreated = true;
    }

    FileIO.append(round + delimiter + edgeCuts + delimiter + numberOfSwaps + delimiter + migrations + "\n",
        outputFilePath);
  }
}
