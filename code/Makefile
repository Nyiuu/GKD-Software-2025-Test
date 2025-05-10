CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra
TARGET = main

HEADERS = model.hpp
all: $(TARGET)

$(TARGET): main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean