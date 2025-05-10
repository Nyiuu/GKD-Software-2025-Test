CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra
# 使用 pkg-config 自动获取 OpenCV 编译和链接标志
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
TARGET = main

HEADERS = model.hpp readphoto.hpp ui.hpp

all: $(TARGET)

$(TARGET): main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean