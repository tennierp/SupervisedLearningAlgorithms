CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2

TARGET = SupervisedLearningAlgorithms
SOURCES = main.cpp
HEADERS = $(wildcard *.h)

TRAIN = mnist_train_10000_features.csv
TEST  = mnist_test_2000_features.csv

all: $(TARGET)

# Rebuild when main.cpp or any header (data.h, naive_bayes.h, ...) changes
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

# Build if needed, then run with the two data files
run: $(TARGET)
	./$(TARGET) $(TRAIN) $(TEST)

clean:
	rm -f $(TARGET)

.PHONY: all run clean