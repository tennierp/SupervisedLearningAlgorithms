CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra

TARGET = SupervisedLearningAlgorithms
SOURCES = main.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean