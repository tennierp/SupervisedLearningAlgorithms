// perceptron.hpp - Multiclass perceptron
//
// Algorithm (from the project description):
//   Input:  training data {(x_i, y_i)}, x_i in R^d, y_i in {0..9}; x_0 = -1 is the bias.
//   Output: one weight vector w(j) per class j, initialized to small random values.
//   For each training example:
//     predict  y' = argmax_k  w(k) . x
//     if y' != y (mistake):
//        w(y)  = w(y)  + eta * x      (correct label)
//        w(y') = w(y') - eta * x      (predicted label)
//
// Training runs for a fixed number of epochs (100). After each epoch the number
// of test errors is recorded, and the weights from the epoch with the fewest
// test errors are kept as the final model.
//
// Features are min-max scaled to [0, 1] (training-set min/max) so that no single
// feature dominates the dot products. The training order is shuffled every epoch
// with a fixed seed, so results are reproducible.
#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include "data.hpp"
#include <algorithm>
#include <numeric>
#include <random>

class Perceptron {
public:
    double eta = 0.1;              // learning rate
    int epochs = 100;
    bool normalize = true;
    unsigned seed = 42;

    // Filled in by train():
    std::vector<int> testErrorsPerEpoch;    // test errors after epoch 1..epochs
    std::vector<int> trainErrorsPerEpoch;   // mistakes made during epoch 1..epochs
    int bestEpoch = 0;                      // 1-based epoch with fewest test errors
    int bestErrors = 0;

    void train(const Dataset& d, const Dataset& test) {
        int F = d.numFeatures();
        computeScaling(d);
        std::vector<std::vector<double>> Xtr = scaleAll(d), Xte = scaleAll(test);

        // Small random initial weights; index 0 is the bias weight (x_0 = -1).
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> small(-0.01, 0.01);
        W.assign(NUM_CLASSES, std::vector<double>(F + 1));
        for (auto& w : W) for (double& v : w) v = small(rng);

        std::vector<int> order(d.size());
        std::iota(order.begin(), order.end(), 0);
        testErrorsPerEpoch.clear();
        trainErrorsPerEpoch.clear();
        bestErrors = test.size() + 1;
        std::vector<std::vector<double>> bestW = W;

        for (int e = 1; e <= epochs; e++) {
            std::shuffle(order.begin(), order.end(), rng);
            int mistakes = 0;
            for (int i : order) {
                const std::vector<double>& x = Xtr[i];
                int pred = argmax(x), actual = d.y[i];
                if (pred != actual) {
                    mistakes++;
                    W[actual][0] -= eta;           // bias input is -1
                    W[pred][0]   += eta;
                    for (int f = 0; f < F; f++) {
                        W[actual][f + 1] += eta * x[f];
                        W[pred][f + 1]   -= eta * x[f];
                    }
                }
            }
            trainErrorsPerEpoch.push_back(mistakes);

            int errors = 0;
            for (int i = 0; i < test.size(); i++)
                if (argmax(Xte[i]) != test.y[i]) errors++;
            testErrorsPerEpoch.push_back(errors);
            if (errors < bestErrors) { bestErrors = errors; bestEpoch = e; bestW = W; }
        }
        W = bestW;                                 // keep the best epoch's weights
    }

    int predict(const std::vector<double>& rawX) const {
        return argmax(scale(rawX));
    }

    std::vector<int> predictAll(const Dataset& d) const {
        std::vector<int> p(d.size());
        for (int i = 0; i < d.size(); i++) p[i] = predict(d.X[i]);
        return p;
    }

private:
    std::vector<std::vector<double>> W;   // W[class][0] = bias weight, W[class][f+1] = feature f
    std::vector<double> minVal, range;

    // argmax_k  w(k) . x   with x_0 = -1
    int argmax(const std::vector<double>& x) const {
        int best = 0;
        double bestScore = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            double s = -W[c][0];
            for (size_t f = 0; f < x.size(); f++) s += W[c][f + 1] * x[f];
            if (c == 0 || s > bestScore) { bestScore = s; best = c; }
        }
        return best;
    }

    void computeScaling(const Dataset& d) {
        int F = d.numFeatures();
        minVal.assign(F, 0.0);
        range.assign(F, 1.0);
        if (!normalize) return;
        for (int f = 0; f < F; f++) {
            double lo = d.X[0][f], hi = d.X[0][f];
            for (int i = 1; i < d.size(); i++) {
                lo = std::min(lo, d.X[i][f]);
                hi = std::max(hi, d.X[i][f]);
            }
            minVal[f] = lo;
            range[f] = (hi > lo) ? hi - lo : 1.0;
        }
    }

    std::vector<double> scale(const std::vector<double>& x) const {
        std::vector<double> s(x.size());
        for (size_t f = 0; f < x.size(); f++) s[f] = (x[f] - minVal[f]) / range[f];
        return s;
    }

    std::vector<std::vector<double>> scaleAll(const Dataset& d) const {
        std::vector<std::vector<double>> out;
        out.reserve(d.size());
        for (const auto& row : d.X) out.push_back(scale(row));
        return out;
    }
};

#endif