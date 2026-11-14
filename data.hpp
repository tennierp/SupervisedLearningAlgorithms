// data.hpp - shared data loading and evaluation helpers for CS 480 Project 4
#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>

const int NUM_CLASSES = 10;

struct Dataset {
    std::vector<std::string> featureNames;     // 20 feature names
    std::vector<std::vector<double>> X;        // X[i][f]
    std::vector<int> y;                        // labels 0..9
    int numFeatures() const { return (int)featureNames.size(); }
    int size() const { return (int)X.size(); }
};

// Reads a CSV whose header is f1,...,f20,label and whose last column is the label.
inline Dataset loadCSV(const std::string& path) {
    Dataset d;
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: cannot open " << path << "\n";
        std::exit(1);
    }
    std::string line;
    std::getline(in, line);                    // header
    {
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) d.featureNames.push_back(tok);
        d.featureNames.pop_back();             // drop "label"
    }
    while (std::getline(in, line)) {
        if (line.empty() || line == "\r") continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<double> row;
        while (std::getline(ss, tok, ',')) row.push_back(std::atof(tok.c_str()));
        d.y.push_back((int)row.back());
        row.pop_back();
        d.X.push_back(row);
    }
    return d;
}

// Confusion matrix: cm[actual][predicted]
typedef std::vector<std::vector<int>> ConfMatrix;

inline ConfMatrix confusionMatrix(const std::vector<int>& actual,
                                  const std::vector<int>& predicted) {
    ConfMatrix cm(NUM_CLASSES, std::vector<int>(NUM_CLASSES, 0));
    for (size_t i = 0; i < actual.size(); i++) cm[actual[i]][predicted[i]]++;
    return cm;
}

inline double accuracy(const std::vector<int>& actual,
                       const std::vector<int>& predicted) {
    int correct = 0;
    for (size_t i = 0; i < actual.size(); i++)
        if (actual[i] == predicted[i]) correct++;
    return 100.0 * correct / actual.size();
}

inline void printConfusionMatrix(const ConfMatrix& cm) {
    std::cout << "        Predicted ->\n  Act ";
    for (int j = 0; j < NUM_CLASSES; j++) std::cout << std::setw(5) << j;
    std::cout << "\n";
    for (int i = 0; i < NUM_CLASSES; i++) {
        std::cout << std::setw(5) << i << " ";
        for (int j = 0; j < NUM_CLASSES; j++) std::cout << std::setw(5) << cm[i][j];
        std::cout << "\n";
    }
}

#endif