// naive_bayes.h - Naive Bayes classifier
//
// Numerical (continuous) features -> Gaussian likelihood  N(mean_c, var_c)
// Discrete integer features (e.g. loops, *_intersections_max) -> frequency
//   counts with Laplace correction mu:
//       P(x = v | c) = (count(v, c) + mu) / (N_c + mu * V)
//   where V is the number of possible values of that feature.
// Class priors also use Laplace correction:
//       P(c) = (N_c + mu) / (N + mu * K)
// All products are computed as sums of logs to avoid underflow.
#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include "data.h"
#include <cmath>
#include <set>
#include <limits>

class NaiveBayes {
public:
    double mu = 1.0;                      // Laplace correction parameter

    void train(const Dataset& d) {
        int F = d.numFeatures(), N = d.size();
        detectDiscrete(d);

        classCount.assign(NUM_CLASSES, 0);
        for (int i = 0; i < N; i++) classCount[d.y[i]]++;
        totalCount = N;

        // ---- Gaussian parameters: mean and variance per class per feature
        mean.assign(NUM_CLASSES, std::vector<double>(F, 0.0));
        var.assign(NUM_CLASSES, std::vector<double>(F, 0.0));
        for (int i = 0; i < N; i++)
            for (int f = 0; f < F; f++) mean[d.y[i]][f] += d.X[i][f];
        for (int c = 0; c < NUM_CLASSES; c++)
            for (int f = 0; f < F; f++) mean[c][f] /= classCount[c];
        for (int i = 0; i < N; i++)
            for (int f = 0; f < F; f++) {
                double diff = d.X[i][f] - mean[d.y[i]][f];
                var[d.y[i]][f] += diff * diff;
            }
        for (int c = 0; c < NUM_CLASSES; c++)
            for (int f = 0; f < F; f++) {
                var[c][f] /= classCount[c];
                if (var[c][f] < VAR_FLOOR) var[c][f] = VAR_FLOOR;   // avoid divide-by-0
            }

        // ---- Discrete counts: counts[f][c][v]
        counts.assign(F, std::vector<std::vector<int>>());
        for (int f = 0; f < F; f++) {
            if (!isDiscrete[f]) continue;
            counts[f].assign(NUM_CLASSES, std::vector<int>(numValues[f], 0));
            for (int i = 0; i < N; i++) {
                int v = (int)std::lround(d.X[i][f]);
                counts[f][d.y[i]][v]++;
            }
        }
    }

    int predict(const std::vector<double>& x) const {
        int best = 0;
        double bestScore = -std::numeric_limits<double>::infinity();
        for (int c = 0; c < NUM_CLASSES; c++) {
            double s = logSafe((classCount[c] + mu) / (totalCount + mu * NUM_CLASSES));
            for (size_t f = 0; f < x.size(); f++) {
                if (isDiscrete[f]) {
                    int v = (int)std::lround(x[f]);
                    int cnt = (v >= 0 && v < numValues[f]) ? counts[f][c][v] : 0;
                    s += logSafe((cnt + mu) / (classCount[c] + mu * numValues[f]));
                } else {
                    double diff = x[f] - mean[c][f];
                    s += -0.5 * std::log(2 * M_PI * var[c][f])
                         - diff * diff / (2 * var[c][f]);
                }
            }
            if (s > bestScore) { bestScore = s; best = c; }
        }
        return best;
    }

    std::vector<int> predictAll(const Dataset& d) const {
        std::vector<int> p(d.size());
        for (int i = 0; i < d.size(); i++) p[i] = predict(d.X[i]);
        return p;
    }

    void printFeatureTypes(const Dataset& d) const {
        std::cout << "Discrete features (Laplace-corrected counts): ";
        for (int f = 0; f < d.numFeatures(); f++)
            if (isDiscrete[f]) std::cout << d.featureNames[f] << "  ";
        std::cout << "\nAll other features use Gaussian likelihoods.\n";
    }

private:
    static constexpr double VAR_FLOOR = 1e-9;
    static constexpr int MAX_DISCRETE_VALUES = 15;

    std::vector<int> classCount;
    int totalCount = 0;
    std::vector<std::vector<double>> mean, var;
    std::vector<bool> isDiscrete;
    std::vector<int> numValues;                         // V for each discrete feature
    std::vector<std::vector<std::vector<int>>> counts;  // counts[f][c][v]

    // log() that tolerates probability 0 (happens when mu = 0 and a value was unseen)
    static double logSafe(double p) {
        return p > 0 ? std::log(p) : -1e9;
    }

    // A feature is discrete if every value is a small non-negative integer.
    void detectDiscrete(const Dataset& d) {
        int F = d.numFeatures();
        isDiscrete.assign(F, true);
        numValues.assign(F, 0);
        for (int f = 0; f < F; f++) {
            std::set<int> vals;
            for (int i = 0; i < d.size() && isDiscrete[f]; i++) {
                double v = d.X[i][f];
                if (v < 0 || std::fabs(v - std::round(v)) > 1e-9) isDiscrete[f] = false;
                else vals.insert((int)std::lround(v));
            }
            if (isDiscrete[f] && (int)vals.size() > MAX_DISCRETE_VALUES) isDiscrete[f] = false;
            if (isDiscrete[f]) numValues[f] = *vals.rbegin() + 1;   // values 0..max
        }
    }
};

#endif