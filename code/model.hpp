#pragma once
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>
#include <fstream>
#include "readphoto.hpp"
#include <iostream>
using namespace std;
using json = nlohmann::json;

template<typename T>
class Model
{
private:
    vector<vector<T>> weight1;
    vector<vector<T>> bias1;
    vector<vector<T>> weight2;
    vector<vector<T>> bias2;

    vector<vector<T>> matrix_multiply(const vector<vector<T>>& matrix1, const vector<vector<T>>& matrix2){
        int row = matrix1.size();
        int col = matrix2[0].size();
        int w = matrix2.size();

        vector<vector<T>> result(row, vector<T>(col));

        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                T sum = 0;
                for(int k = 0; k < w; k++){
                    sum += matrix1[i][k] * matrix2[k][j];
                }
               result[i][j]= sum;
            }
        }

        return result;
    }

    vector<vector<T>> matrix_add(const vector<vector<T>>& matrix1, const vector<vector<T>>& matrix2){
        vector<vector<T>> result(matrix1.size(), vector<T>(matrix1[0].size()));
        
        for(size_t i = 0; i < matrix1.size(); i++){
            for(size_t j = 0; j < matrix1[0].size(); j++){
                result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }

        return result;
    }

    vector<vector<T>> relu(const vector<vector<T>>& input){
        vector<vector<T>> output(input.size(), vector<T>(input[0].size()));
       
        for(size_t i = 0; i < input.size(); i++){
            for(size_t j = 0; j < input[0].size(); j++){
                output[i][j] = input[i][j] < 0? 0 : input[i][j];
            }
        }

        return output;
    }

    
    vector<T> softMax(const vector<vector<T>>& input){
        int s = input[0].size();
        vector<T> output(s);
        T mother = 0;
        for(int j = 0; j < s; j++){
            mother += exp(input[0][j]);
        }
        for(int j = 0; j < s; j++){
            output[j] = exp(input[0][j]) / mother;
        }

        return output;
    }

    vector<vector<T>> init_matrix(const vector<T>& dims){
        return vector<vector<T>>(
            dims[0], 
            vector<T>(dims[1], 0.0f)
        );
    }

    void load_model(){
        ifstream file("../mnist-fc/meta.json");
        json data = json::parse(file);
        weight1 = init_matrix(data["fc1.weight"].get<vector<T>>());
        bias1 = init_matrix(data["fc1.bias"].get<vector<T>>());
        weight2 = init_matrix(data["fc2.weight"].get<vector<T>>());
        bias2 = init_matrix(data["fc2.bias"].get<vector<T>>());
    }

    void load_data(const string& bin_name, vector<vector<T>>& matrix){
        int row = matrix.size();
        int col = matrix[0].size();

        ifstream f(bin_name, ios::binary);
        if(!f){
            cerr << "can't open file" << endl;
            return;
        }
        //获取文件大小
        f.seekg(0, ios::end);
        auto file_size = f.tellg();
        f.seekg(0, ios::beg);

        auto count = file_size / sizeof(T);

        vector<T> buffer(count);
        f.read(reinterpret_cast<char*>(buffer.data()), file_size);

        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                matrix[i][j] = buffer[i * col + j];
            }
        }
    }

    void load_weight(){
        load_data("../mnist-fc/fc1.weight", weight1);
        load_data("../mnist-fc/fc2.weight", weight2);
        load_data("../mnist-fc/fc1.bias", bias1);
        load_data("../mnist-fc/fc2.bias", bias2);
    }

public:
    
    Model(){
        load_model();
        load_weight();
    }
    
    ~Model(){}

    vector<T> forward(const string& imagePath){
        vector<vector<T>> input = processImage<float>(imagePath);
        vector<T> output;
        auto temp = matrix_multiply(input, weight1);
        temp = matrix_add(temp, bias1);
        temp = relu(temp);
        temp = matrix_multiply(temp, weight2);
        temp = matrix_add(temp, bias2);
        output = softMax(temp);
        return output;
    }

    void process_all(){
        const string folderPath = "../nums/";
        const int numImages = 10;
        for (int i = 0; i < numImages; ++i) {
            string imagePath = folderPath + to_string(i) + ".png"; 
            vector<T> imageData = forward(imagePath);
            
            vector<vector<T>> allImagesData;
    
            if (!imageData.empty()) {
                allImagesData.push_back(imageData);
                cout << "已处理图像: " << imagePath << " - 数据长度: " << imageData.size() << endl;
            }

            if (!allImagesData.empty()) {
                cout << "\n第"+ to_string(i + 1) +"张图片的前10个归一化像素值:" << endl;
                for (int j = 0; j < 10 && j < allImagesData[0].size(); ++j) {
                    cout << allImagesData[0][j] << " ";
                }
                cout << endl;
            }
        }
    }
};
