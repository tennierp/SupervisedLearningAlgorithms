// knn.hpp - k-nearest neighbors classifier
//
// Training just stores the training examples. To classify x, compute the
// Euclidean distance from x to every training example, take the k closest,
// and return the majority label among them.
//
// Feature scaling (optional, on by default): each feature is min-max scaled to
// [0, 1] using the training-set minimum and maximum, so features with larger
// ranges (loops: 0-8, *_intersections_max: 1-5) don't dominate the distance.
//
// Ties in the vote are broken in favor of the tied class whose neighbors have
// the smallest total distance (i.e. the "closer" class).
#ifndef KNN_HPP
#define KNN_HPP

#include "data.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

class KNN {
public:
    int k = 3;
    bool normalize = true;

    void train(const Dataset& d) {
        int F = d.numFeatures();
        minVal.assign(F, 0.0);
        range.assign(F, 1.0);
        if (normalize) {
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
        X.clear();
        for (const auto& row : d.X) X.push_back(scale(row));
        y = d.y;
    }

    // Sorted list of (squared distance, label) for the K nearest training examples.
    std::vector<std::pair<double, int>> neighbors(const std::vector<double>& rawX, int K) const {
        std::vector<double> x = scale(rawX);
        std::vector<std::pair<double, int>> dist(X.size());
        for (size_t i = 0; i < X.size(); i++) {
            double s = 0;
            for (size_t f = 0; f < x.size(); f++) {
                double diff = X[i][f] - x[f];
                s += diff * diff;
            }
            dist[i] = {s, y[i]};
        }
        K = std::min<int>(K, (int)dist.size());
        std::partial_sort(dist.begin(), dist.begin() + K, dist.end());
        dist.resize(K);
        return dist;
    }

    // Majority vote among the first kk entries of a sorted neighbor list.
    // Lets the caller try many k values from a single distance computation.
    static int vote(const std::vector<std::pair<double, int>>& nb, int kk) {
        std::vector<int> count(NUM_CLASSES, 0);
        std::vector<double> distSum(NUM_CLASSES, 0.0);
        for (int i = 0; i < kk && i < (int)nb.size(); i++) {
            count[nb[i].second]++;
            distSum[nb[i].second] += std::sqrt(nb[i].first);
        }
        int best = 0;
        for (int c = 1; c < NUM_CLASSES; c++)
            if (count[c] > count[best] ||
                (count[c] == count[best] && count[c] > 0 && distSum[c] < distSum[best]))
                best = c;
        return best;
    }

    int predict(const std::vector<double>& x) const {
        return vote(neighbors(x, k), k);
    }

    std::vector<int> predictAll(const Dataset& d) const {
        std::vector<int> p(d.size());
        for (int i = 0; i < d.size(); i++) p[i] = predict(d.X[i]);
        return p;
    }

private:
    std::vector<std::vector<double>> X;   // (scaled) training features
    std::vector<int> y;
    std::vector<double> minVal, range;

    std::vector<double> scale(const std::vector<double>& x) const {
        std::vector<double> s(x.size());
        for (size_t f = 0; f < x.size(); f++) s[f] = (x[f] - minVal[f]) / range[f];
        return s;
    }
};

#endif