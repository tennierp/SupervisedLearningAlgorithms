// main.cpp - CS 480 Project 4: Supervised learning on MNIST features
//
// Build: g++ -O2 -std=c++17 main.cpp -o project4
// Run:   ./project4 [train.csv] [test.csv]
//        (defaults: mnist_train_10000_features.csv mnist_test_2000_features.csv)
//
// Menu:
//   1) Summary: accuracy (%), confusion matrix and CPU time for each algorithm
//   2) Enter j (1..12000): show the true label and each algorithm's prediction.
//      1..10000 = training instances, 10001..12000 = test instances.
#include "data.hpp"
#include "naive_bayes.hpp"
#include "decision_tree.hpp"
#include "knn.hpp"
#include "perceptron.hpp"
#include <ctime>
#include <limits>

// Results stored for each algorithm so the summary can be printed at any time.
struct AlgoResult {
    std::string name;
    std::string settings;        // chosen hyper-parameter(s), e.g. "mu = 0.00"
    std::string experiments;     // table of every setting tried and its accuracy
    double accuracy = 0;
    double cpuSeconds = 0;       // training + predicting the test set
    ConfMatrix cm;
};

static double cpuSince(clock_t start) {
    return double(clock() - start) / CLOCKS_PER_SEC;
}

// ---------------------------------------------------------------- Naive Bayes
// Tries mu = 0.00, 0.05, ..., 1.00 in order. Ties are allowed; as soon as accuracy
// drops below the best so far, the search stops and the best value is kept.
static AlgoResult runNaiveBayes(NaiveBayes& nb, const Dataset& train, const Dataset& test) {
    double bestMu = 0, bestAcc = -1;
    std::ostringstream table;
    table << std::fixed << std::setprecision(2) << "    mu    accuracy\n";
    for (int step = 0; step <= 20; step++) {
        nb.mu = step * 0.05;
        nb.train(train);
        double acc = accuracy(test.y, nb.predictAll(test));
        table << "  " << std::setw(5) << nb.mu << "    " << acc << "%\n";
        if (acc < bestAcc) {                       // accuracy dropped -> stop
            table << "  (accuracy dropped - search ended)\n";
            break;
        }
        if (acc > bestAcc) {                       // a tie keeps the earlier value
            bestAcc = acc;
            bestMu = nb.mu;
        }
    }

    // Final model with the chosen mu; this run is the one that's timed.
    nb.mu = bestMu;
    clock_t start = clock();
    nb.train(train);
    std::vector<int> pred = nb.predictAll(test);

    AlgoResult r;
    r.cpuSeconds = cpuSince(start);
    r.name = "Naive Bayes";
    std::ostringstream s;
    s << std::fixed << std::setprecision(2) << "mu = " << bestMu;
    r.settings = s.str();
    r.experiments = table.str();
    r.accuracy = accuracy(test.y, pred);
    r.cm = confusionMatrix(test.y, pred);
    return r;
}

// ---------------------------------------------------------------- Decision tree
// Tries all three impurity measures. For each measure, the maximum depth is
// increased step by step until accuracy drops (ties continue the search); the best
// depth for that measure is the most accurate one seen before the drop. The measure/depth pair with the
// lowest test error (highest accuracy) overall is kept.
static AlgoResult runDecisionTree(DecisionTree& dt, const Dataset& train, const Dataset& test) {
    const Impurity measures[] = {Impurity::Gini, Impurity::Misclassification, Impurity::Entropy};
    const int depths[] = {4, 6, 8, 10, 12, 14, 16, 20, 1000};   // 1000 = no limit
    Impurity bestMeasure = Impurity::Gini;
    int bestDepth = 1000;
    double bestAcc = -1;

    std::ostringstream table;
    table << std::fixed << std::setprecision(2)
          << "  " << std::left << std::setw(26) << "Measure" << std::right
          << std::setw(10) << "max depth" << std::setw(8) << "nodes" << std::setw(11) << "accuracy\n";
    for (Impurity m : measures) {
        double measureBestAcc = -1;
        int measureBestDepth = depths[0];
        for (int depth : depths) {
            dt.measure = m;
            dt.maxDepth = depth;
            dt.train(train);
            double acc = accuracy(test.y, dt.predictAll(test));
            table << "  " << std::left << std::setw(26) << impurityName(m) << std::right
                  << std::setw(10) << (depth == 1000 ? std::string("none") : std::to_string(depth))
                  << std::setw(8) << dt.nodeCount() << std::setw(9) << acc << "%\n";
            if (acc < measureBestAcc) {            // accuracy dropped -> stop
                table << "  (accuracy dropped - " << impurityName(m) << " search ended)\n";
                break;
            }
            if (acc > measureBestAcc) {            // a tie keeps the smaller depth
                measureBestAcc = acc;
                measureBestDepth = depth;
            }
        }
        if (measureBestAcc > bestAcc) {
            bestAcc = measureBestAcc;
            bestMeasure = m;
            bestDepth = measureBestDepth;
        }
    }

    dt.measure = bestMeasure;
    dt.maxDepth = bestDepth;
    clock_t start = clock();
    dt.train(train);
    std::vector<int> pred = dt.predictAll(test);

    AlgoResult r;
    r.cpuSeconds = cpuSince(start);
    r.name = "Decision Tree";
    r.settings = impurityName(bestMeasure) + ", max depth "
               + (bestDepth == 1000 ? std::string("none") : std::to_string(bestDepth))
               + ", " + std::to_string(dt.nodeCount()) + " nodes";
    r.experiments = table.str();
    r.accuracy = accuracy(test.y, pred);
    r.cm = confusionMatrix(test.y, pred);
    return r;
}

// ---------------------------------------------------------------- k-NN
// Tries increasing k, with and without min-max feature scaling. For each scaling
// option, k grows until accuracy drops (ties continue the search); the best k is
// the most accurate one seen before the drop. For speed, each test example's neighbor list (up to the largest k)
// is computed once and reused for every k.
static AlgoResult runKNN(KNN& knn, const Dataset& train, const Dataset& test) {
    const int ks[] = {1, 3, 5, 7, 9, 11, 15, 21, 31};
    const int numK = sizeof(ks) / sizeof(ks[0]);
    const int maxK = ks[numK - 1];
    int bestK = 1;
    bool bestNorm = true;
    double bestAcc = -1;

    std::ostringstream table;
    table << std::fixed << std::setprecision(2) << "  features     k  accuracy\n";
    for (int norm = 1; norm >= 0; norm--) {
        knn.normalize = (norm == 1);
        knn.train(train);

        // Neighbor lists for every test example, computed once (the expensive part).
        std::vector<std::vector<std::pair<double, int>>> nbLists(test.size());
        for (int i = 0; i < test.size(); i++) nbLists[i] = knn.neighbors(test.X[i], maxK);

        double normBestAcc = -1;
        int normBestK = ks[0];
        for (int j = 0; j < numK; j++) {
            int correct = 0;
            for (int i = 0; i < test.size(); i++)
                if (KNN::vote(nbLists[i], ks[j]) == test.y[i]) correct++;
            double acc = 100.0 * correct / test.size();
            table << std::setw(10) << (norm ? "scaled" : "raw") << std::setw(6) << ks[j]
                  << std::setw(9) << acc << "%\n";
            if (acc < normBestAcc) {               // accuracy dropped -> stop
                table << "  (accuracy dropped - search ended)\n";
                break;
            }
            if (acc > normBestAcc) {               // a tie keeps the smaller k
                normBestAcc = acc;
                normBestK = ks[j];
            }
        }
        if (normBestAcc > bestAcc) {
            bestAcc = normBestAcc;
            bestK = normBestK;
            bestNorm = knn.normalize;
        }
    }

    knn.k = bestK;
    knn.normalize = bestNorm;
    clock_t start = clock();
    knn.train(train);
    std::vector<int> pred = knn.predictAll(test);

    AlgoResult r;
    r.cpuSeconds = cpuSince(start);
    r.name = "k-NN";
    r.settings = "k = " + std::to_string(bestK) + (bestNorm ? ", scaled features" : ", raw features");
    r.experiments = table.str();
    r.accuracy = accuracy(test.y, pred);
    r.cm = confusionMatrix(test.y, pred);
    return r;
}

// ---------------------------------------------------------------- Perceptron
// Tries increasing learning rates eta, with and without feature scaling. For each
// scaling option, eta grows until the test errors go up (ties continue the search);
// the best eta is the most accurate one seen before that. Every run trains for the full 100 epochs required by the
// spec and keeps the weights from the epoch with the fewest test errors.
static AlgoResult runPerceptron(Perceptron& pc, const Dataset& train, const Dataset& test) {
    const double etas[] = {0.01, 0.1, 1.0};
    double bestEta = 0.1;
    bool bestNorm = true;
    int fewest = test.size() + 1;

    std::ostringstream table;
    table << std::fixed << std::setprecision(2)
          << "   eta   features  best epoch  test errors  accuracy\n";
    for (int norm = 1; norm >= 0; norm--) {
        int normFewest = test.size() + 1;
        double normBestEta = etas[0];
        for (double eta : etas) {
            pc.eta = eta;
            pc.normalize = (norm == 1);
            pc.train(train, test);
            table << std::setw(6) << eta << std::setw(10) << (norm ? "scaled" : "raw")
                  << std::setw(12) << pc.bestEpoch << std::setw(13) << pc.bestErrors
                  << std::setw(9) << 100.0 * (test.size() - pc.bestErrors) / test.size() << "%\n";
            if (pc.bestErrors > normFewest) {       // more errors = accuracy dropped -> stop
                table << "  (accuracy dropped - search ended)\n";
                break;
            }
            if (pc.bestErrors < normFewest) {       // a tie keeps the smaller eta
                normFewest = pc.bestErrors;
                normBestEta = eta;
            }
        }
        if (normFewest < fewest) {
            fewest = normFewest;
            bestEta = normBestEta;
            bestNorm = (norm == 1);
        }
    }

    pc.eta = bestEta;
    pc.normalize = bestNorm;
    clock_t start = clock();
    pc.train(train, test);
    std::vector<int> pred = pc.predictAll(test);

    AlgoResult r;
    r.cpuSeconds = cpuSince(start);

    // Test errors after every epoch of the chosen run, 10 epochs per row.
    table << "\n  Test errors per epoch (eta = " << bestEta
          << (bestNorm ? ", scaled" : ", raw") << "):\n  epoch  ";
    for (int c = 0; c < 10; c++) table << std::setw(6) << "+" + std::to_string(c);
    table << "\n";
    for (int row = 1; row <= pc.epochs; row += 10) {       // row starts at epoch 1, 11, 21, ...
        table << "  " << std::setw(5) << row << "  ";
        for (int c = 0; c < 10 && row + c <= pc.epochs; c++)
            table << std::setw(6) << pc.testErrorsPerEpoch[row + c - 1];
        table << "\n";
    }
    table << "  Minimum: " << pc.bestErrors << " test errors at epoch " << pc.bestEpoch << "\n";

    r.name = "Perceptron";
    std::ostringstream st;
    st << "eta = " << bestEta << (bestNorm ? ", scaled features" : ", raw features")
       << ", best epoch " << pc.bestEpoch << " of " << pc.epochs;
    r.settings = st.str();
    r.experiments = table.str();
    r.accuracy = accuracy(test.y, pred);
    r.cm = confusionMatrix(test.y, pred);
    return r;
}

// ---------------------------------------------------------------- menu actions
static void printSummary(const std::vector<AlgoResult>& results) {
    std::cout << "\n================ SUMMARY (test set) ================\n";
    for (const AlgoResult& r : results) {
        std::cout << "\n--- " << r.name << " (" << r.settings << ") ---\n"
                  << "Settings tried:\n" << r.experiments
                  << std::fixed << std::setprecision(2)
                  << "Accuracy: " << r.accuracy << "%   CPU time: "
                  << std::setprecision(4) << r.cpuSeconds << " s\n";
        printConfusionMatrix(r.cm);
    }
    std::cout << "\n" << std::left << std::setw(16) << "Algorithm"
              << std::setw(14) << "Accuracy" << "CPU time (s)\n" << std::right;
    for (const AlgoResult& r : results)
        std::cout << std::left << std::setw(16) << r.name << std::right
                  << std::fixed << std::setprecision(2) << std::setw(7) << r.accuracy
                  << "%      " << std::setprecision(4) << r.cpuSeconds << "\n";
}

static void classifyInstance(int j, const Dataset& train, const Dataset& test,
                             const NaiveBayes& nb, const DecisionTree& dt, const KNN& knn,
                             const Perceptron& pc) {
    bool isTrain = j <= train.size();
    const Dataset& d = isTrain ? train : test;
    int idx = isTrain ? j - 1 : j - train.size() - 1;
    const std::vector<double>& x = d.X[idx];

    std::cout << "\nInstance " << j << " ("
              << (isTrain ? "training" : "test") << " #" << idx + 1 << ")\n"
              << "Correct label:  " << d.y[idx] << "\n"
              << "Naive Bayes:    " << nb.predict(x) << "\n"
              << "Decision Tree:  " << dt.predict(x) << "\n"
              << "k-NN:           " << knn.predict(x) << "\n"
              << "Perceptron:     " << pc.predict(x) << "\n";
}

static int readInt(const std::string& prompt) {
    int v;
    while (true) {
        std::cout << prompt;
        if (std::cin >> v) return v;
        if (std::cin.eof()) std::exit(0);
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Please enter a whole number.\n";
    }
}

// ---------------------------------------------------------------- main
int main(int argc, char* argv[]) {
    std::string trainPath = argc > 1 ? argv[1] : "mnist_train_10000_features.csv";
    std::string testPath  = argc > 2 ? argv[2] : "mnist_test_2000_features.csv";
    Dataset train = loadCSV(trainPath), test = loadCSV(testPath);
    int total = train.size() + test.size();
    std::cout << "Loaded " << train.size() << " training and " << test.size()
              << " test examples (" << train.numFeatures() << " features).\n"
              << "Training all algorithms...\n";

    std::vector<AlgoResult> results;
    NaiveBayes nb;
    results.push_back(runNaiveBayes(nb, train, test));
    DecisionTree dt;
    results.push_back(runDecisionTree(dt, train, test));
    KNN knn;
    results.push_back(runKNN(knn, train, test));
    Perceptron pc;
    results.push_back(runPerceptron(pc, train, test));
    std::cout << "Done.\n";

    while (true) {
        std::cout << "\n=========== MENU ===========\n"
                  << "1) Summary of results\n"
                  << "2) Classify instance j (1-" << total << ")\n"
                  << "0) Quit\n";
        int choice = readInt("Choice: ");
        if (choice == 0) break;
        if (choice == 1) {
            printSummary(results);
        } else if (choice == 2) {
            int j = readInt("Enter j (1-" + std::to_string(total) + "): ");
            if (j < 1 || j > total) std::cout << "j must be between 1 and " << total << ".\n";
            else classifyInstance(j, train, test, nb, dt, knn, pc);
        } else {
            std::cout << "Invalid choice.\n";
        }
    }
    return 0;
}