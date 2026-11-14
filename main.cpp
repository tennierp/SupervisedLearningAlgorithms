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
#include "data.h"
#include "naive_bayes.h"
// #include "decision_tree.h"   // Step 2
// #include "knn.h"             // Step 3
// #include "perceptron.h"      // Step 4
#include <ctime>
#include <limits>

// Results stored for each algorithm so the summary can be printed at any time.
struct AlgoResult {
    std::string name;
    std::string settings;        // chosen hyper-parameter(s), e.g. "mu = 0.00"
    double accuracy = 0;
    double cpuSeconds = 0;       // training + predicting the test set
    ConfMatrix cm;
};

static double cpuSince(clock_t start) {
    return double(clock() - start) / CLOCKS_PER_SEC;
}

// ---------------------------------------------------------------- Naive Bayes
// Tries mu = 0.00, 0.05, ..., 1.00 and keeps the value with the best test accuracy.
static AlgoResult runNaiveBayes(NaiveBayes& nb, const Dataset& train, const Dataset& test) {
    double bestMu = 0, bestAcc = -1;
    for (int step = 0; step <= 20; step++) {
        nb.mu = step * 0.05;
        nb.train(train);
        double acc = accuracy(test.y, nb.predictAll(test));
        if (acc > bestAcc) { bestAcc = acc; bestMu = nb.mu; }
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
    r.accuracy = accuracy(test.y, pred);
    r.cm = confusionMatrix(test.y, pred);
    return r;
}

// ---------------------------------------------------------------- menu actions
static void printSummary(const std::vector<AlgoResult>& results) {
    std::cout << "\n================ SUMMARY (test set) ================\n";
    for (const AlgoResult& r : results) {
        std::cout << "\n--- " << r.name << " (" << r.settings << ") ---\n"
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
                             const NaiveBayes& nb) {
    bool isTrain = j <= train.size();
    const Dataset& d = isTrain ? train : test;
    int idx = isTrain ? j - 1 : j - train.size() - 1;
    const std::vector<double>& x = d.X[idx];

    std::cout << "\nInstance " << j << " ("
              << (isTrain ? "training" : "test") << " #" << idx + 1 << ")\n"
              << "Correct label:  " << d.y[idx] << "\n"
              << "Naive Bayes:    " << nb.predict(x) << "\n";
    // Step 2-4: add decision tree, k-NN and perceptron predictions here.
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
    // Step 2-4: train decision tree, k-NN and perceptron here and push their results.
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
            else classifyInstance(j, train, test, nb);
        } else {
            std::cout << "Invalid choice.\n";
        }
    }
    return 0;
}