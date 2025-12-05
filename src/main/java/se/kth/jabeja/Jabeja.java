package se.kth.jabeja;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.log4j.Logger;

import se.kth.jabeja.config.Config;
import se.kth.jabeja.config.NodeSelectionPolicy;
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

      // Apply simulated annealing (Task 2)
      saCoolDown();

      report();
    }
  }

  /**
   * Task 2: Geometric simulated annealing + restart mechanism
   */
  private void saCoolDown() {

    // Geometric cooling: T = T * (1 - delta)
    T = T * (1 - config.getDelta());

    // Restart mechanism: when T is too low, reset to initial
    if (T < 0.1f) {
      T = config.getTemperature(); // usually 1
    }
  }

  /**
   * Sample and swap algorithm for node p
   */
  private void sampleAndSwap(int nodeId) {
    Node partner = null;
    Node nodep = entireGraph.get(nodeId);

    // Try LOCAL or HYBRID local sampling first
    if (config.getNodeSelectionPolicy() == NodeSelectionPolicy.HYBRID
        || config.getNodeSelectionPolicy() == NodeSelectionPolicy.LOCAL) {

      partner = findPartner(nodeId, getNeighbors(nodep));
    }

    // If no partner found, try RANDOM or HYBRID random sampling
    if (partner == null &&
        (config.getNodeSelectionPolicy() == NodeSelectionPolicy.HYBRID
            || config.getNodeSelectionPolicy() == NodeSelectionPolicy.RANDOM)) {

      partner = findPartner(nodeId, getSample(nodeId));
    }

    // Perform swap if beneficial (partner != null)
    if (partner != null) {
      int pColor = nodep.getColor();
      int qColor = partner.getColor();

      nodep.setColor(qColor);
      partner.setColor(pColor);

      numberOfSwaps++;
    }
  }

  /**
   * Task 1 + Task 2 + BONUS:
   * Partner selection using JaBeJa’s utility + Metropolis probabilistic
   * acceptance.
   */
  public Node findPartner(int nodeId, Integer[] nodes) {

    Node nodep = entireGraph.get(nodeId);
    Node bestPartner = null;

    double highestNewUtility = Double.NEGATIVE_INFINITY;
    double alpha = config.getAlpha();

    for (Integer candidateId : nodes) {
      Node nodeq = entireGraph.get(candidateId);

      // Old state degrees
      int d_pp = getDegree(nodep, nodep.getColor());
      int d_qq = getDegree(nodeq, nodeq.getColor());

      // New state degrees (if swapped)
      int d_pq = getDegree(nodep, nodeq.getColor());
      int d_qp = getDegree(nodeq, nodep.getColor());

      // Compute utilities
      double oldUtility = Math.pow(d_pp, alpha) + Math.pow(d_qq, alpha);
      double newUtility = Math.pow(d_pq, alpha) + Math.pow(d_qp, alpha);

      // BONUS: Metropolis acceptance function
      boolean accept = false;
      double delta = newUtility - oldUtility;

      if (delta > 0) {
        // Always accept improving moves
        accept = true;
      } else {
        // Accept worsening moves with probability exp(delta / T)
        double prob = Math.exp(delta / T); // delta < 0 => prob < 1
        double r = Math.random();
        if (r < prob) {
          accept = true;
        }
      }

      // Select best among accepted candidates
      if (accept && newUtility > highestNewUtility) {
        highestNewUtility = newUtility;
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
   * Uniform random sample of the graph
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

      if (nodeColor != node.getInitColor())
        migrations++;

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
