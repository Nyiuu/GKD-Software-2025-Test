# Makefile for front and back programs (all files in same directory)

# Compiler settings
CXX      := g++
CXXFLAGS := -Wall -Wextra -std=c++20 -pthread
LDFLAGS  := 

# OpenCV settings (using pkg-config)
OPENCV_FLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS  := $(shell pkg-config --libs opencv4)

# JSON library (header-only, no linking needed)
# If nlohmann/json.hpp is in the same folder, no need for special flags

# Targets
TARGETS := front back

# Source files
SRC_front := front.cpp
SRC_back := back.cpp

# Header files in same directory
HEADERS := model.hpp modelbase.hpp

# Default target
all: $(TARGETS)

# Frontend program (needs OpenCV)
front: $(SRC_front) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OPENCV_FLAGS) $(SRC_front) -o $@ $(LDFLAGS) $(OPENCV_LIBS)

# Backend program (needs OpenCV and includes JSON)
back: $(SRC_back) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OPENCV_FLAGS) $(SRC_back) -o $@ $(LDFLAGS) $(OPENCV_LIBS)

# Clean up
clean:
	rm -f $(TARGETS)

.PHONY: all clean