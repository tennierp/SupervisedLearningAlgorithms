// decision_tree.hpp - Binary decision tree classifier
//
// Every internal node asks "Is x[I] <= c ?" for a feature I and constant c.
//   yes -> left child,  no -> right child.
// Candidate thresholds c are the midpoints between consecutive distinct values
// of a feature among the examples that reach the node.
//
// The split chosen at each node is the one that most reduces non-homogeneity
// (impurity), measured by one of:
//   Gini index:               1 - sum_k p_k^2
//   Misclassification error:  1 - max_k p_k
//   Entropy:                  -sum_k p_k log2 p_k
// using the weighted impurity of the two children:
//   (n_left/n) * I(left) + (n_right/n) * I(right)
//
// A node becomes a leaf (predicting its majority class) when it is pure,
// has fewer than minSamplesSplit examples, reaches maxDepth, or no split
// lowers the impurity.
#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include "data.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

enum class Impurity { Gini, Misclassification, Entropy };

inline std::string impurityName(Impurity m) {
    switch (m) {
        case Impurity::Gini:              return "Gini index";
        case Impurity::Misclassification: return "Misclassification error";
        default:                          return "Entropy";
    }
}

class DecisionTree {
public:
    Impurity measure = Impurity::Gini;
    int maxDepth = 1000;          // effectively unlimited
    int minSamplesSplit = 2;

    void train(const Dataset& d) {
        data = &d;
        nodes.clear();
        std::vector<int> idx(d.size());
        std::iota(idx.begin(), idx.end(), 0);
        build(idx, 0);
        data = nullptr;
    }

    int predict(const std::vector<double>& x) const {
        int n = 0;
        while (!nodes[n].isLeaf)
            n = (x[nodes[n].feature] <= nodes[n].threshold) ? nodes[n].left : nodes[n].right;
        return nodes[n].label;
    }

    std::vector<int> predictAll(const Dataset& d) const {
        std::vector<int> p(d.size());
        for (int i = 0; i < d.size(); i++) p[i] = predict(d.X[i]);
        return p;
    }

    int nodeCount() const { return (int)nodes.size(); }

    int leafCount() const {
        int c = 0;
        for (const Node& n : nodes) if (n.isLeaf) c++;
        return c;
    }

    int depth() const { return depthFrom(0); }

private:
    struct Node {
        bool isLeaf = true;
        int label = 0;            // majority class (used when leaf)
        int feature = -1;
        double threshold = 0;
        int left = -1, right = -1;
    };

    std::vector<Node> nodes;
    const Dataset* data = nullptr;

    double impurity(const std::vector<int>& counts, int total) const {
        if (total == 0) return 0;
        double result = 0;
        switch (measure) {
            case Impurity::Gini: {
                double sumSq = 0;
                for (int c : counts) { double p = double(c) / total; sumSq += p * p; }
                result = 1 - sumSq;
                break;
            }
            case Impurity::Misclassification: {
                int mx = *std::max_element(counts.begin(), counts.end());
                result = 1 - double(mx) / total;
                break;
            }
            case Impurity::Entropy: {
                for (int c : counts)
                    if (c > 0) { double p = double(c) / total; result -= p * std::log2(p); }
                break;
            }
        }
        return result;
    }

    // Builds the subtree for the examples in idx; returns the new node's index.
    int build(std::vector<int>& idx, int depth) {
        int id = (int)nodes.size();
        nodes.push_back(Node());
        int n = (int)idx.size();

        std::vector<int> counts(NUM_CLASSES, 0);
        for (int i : idx) counts[data->y[i]]++;
        nodes[id].label = int(std::max_element(counts.begin(), counts.end()) - counts.begin());

        double parentImp = impurity(counts, n);
        if (parentImp == 0 || n < minSamplesSplit || depth >= maxDepth) return id;

        // ---- search every feature and every threshold for the best split
        int bestFeature = -1;
        double bestThreshold = 0, bestImp = parentImp - 1e-12;   // must strictly improve
        std::vector<int> sorted = idx;
        for (int f = 0; f < data->numFeatures(); f++) {
            std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
                return data->X[a][f] < data->X[b][f];
            });
            std::vector<int> leftCounts(NUM_CLASSES, 0), rightCounts = counts;
            for (int k = 0; k < n - 1; k++) {
                int lbl = data->y[sorted[k]];
                leftCounts[lbl]++;
                rightCounts[lbl]--;
                double v = data->X[sorted[k]][f], vNext = data->X[sorted[k + 1]][f];
                if (v == vNext) continue;                 // can only split between distinct values
                int nl = k + 1, nr = n - nl;
                double imp = (nl * impurity(leftCounts, nl) + nr * impurity(rightCounts, nr)) / n;
                if (imp < bestImp) {
                    bestImp = imp;
                    bestFeature = f;
                    bestThreshold = (v + vNext) / 2;
                }
            }
        }
        if (bestFeature < 0) return id;                   // no split helps -> leaf

        std::vector<int> leftIdx, rightIdx;
        for (int i : idx)
            (data->X[i][bestFeature] <= bestThreshold ? leftIdx : rightIdx).push_back(i);
        idx.clear(); idx.shrink_to_fit();                 // free memory before recursing

        nodes[id].isLeaf = false;
        nodes[id].feature = bestFeature;
        nodes[id].threshold = bestThreshold;
        int l = build(leftIdx, depth + 1);
        int r = build(rightIdx, depth + 1);
        nodes[id].left = l;                               // set after recursion: nodes may reallocate
        nodes[id].right = r;
        return id;
    }

    int depthFrom(int n) const {
        if (nodes[n].isLeaf) return 0;
        return 1 + std::max(depthFrom(nodes[n].left), depthFrom(nodes[n].right));
    }
};

#endif