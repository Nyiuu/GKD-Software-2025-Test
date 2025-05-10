#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

template<typename T>
vector<vector<T>> processImage(const string& imagePath) {
    // 1. 以灰度模式读取图像
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    
    if (image.empty()) {
        cerr << "无法加载图像: " << imagePath << endl;
        return {};
    }

    // 2. 缩小图像到28x28像素
    Mat resizedImage;
    resize(image, resizedImage, Size(28, 28), 0, 0, INTER_LINEAR);

    // 3. 将图像拍扁成一维向量
    Mat flattened = resizedImage.reshape(1, 1); // 1通道，1行
    flattened.convertTo(flattened, CV_32F);     // 转换为浮点型

    // 4. 归一化到0-1范围
    vector<T> normalizedVec;
    flattened /= 255.0;
    normalizedVec.assign(flattened.begin<T>(), flattened.end<T>());
    vector<vector<T>> result;
    result.push_back(normalizedVec);
    return result;
}


