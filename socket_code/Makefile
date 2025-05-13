# Makefile for demo1 and demo2 front programs

# Compiler settings
CXX      := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -pthread
LDFLAGS  := 

# Targets
TARGETS :=  front back

# Source files
 
SRC_front := front.cpp
SRC_back := back.cpp

# Default target (build all)
all: $(TARGETS)

front: $(SRC_front) 
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

back: $(SRC_back)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
# Clean up
clean:
	rm -f $(TARGETS)

.PHONY: all clean