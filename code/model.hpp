#pragma once
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>
#include <fstream>
#include "readphoto.hpp"
#include <iostream>
using namespace std;
using json = nlohmann::json;

class Model
{
private:
    vector<vector<float>> weight1;
    vector<vector<float>> bias1;
    vector<vector<float>> weight2;
    vector<vector<float>> bias2;

    vector<vector<float>> matrix_multiply(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2){
        int row = matrix1.size();
        int col = matrix2[0].size();
        int w = matrix2.size();

        vector<vector<float>> result(row, vector<float>(col));

        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                float sum = 0;
                for(int k = 0; k < w; k++){
                    sum += matrix1[i][k] * matrix2[k][j];
                }
               result[i][j]= sum;
            }
        }

        return result;
    }

    vector<vector<float>> matrix_add(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2){
        vector<vector<float>> result(matrix1.size(), vector<float>(matrix1[0].size()));
        
        for(size_t i = 0; i < matrix1.size(); i++){
            for(size_t j = 0; j < matrix1[0].size(); j++){
                result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }

        return result;
    }

    vector<vector<float>> relu(const vector<vector<float>>& input){
        vector<vector<float>> output(input.size(), vector<float>(input[0].size()));
       
        for(size_t i = 0; i < input.size(); i++){
            for(size_t j = 0; j < input[0].size(); j++){
                output[i][j] = input[i][j] < 0? 0 : input[i][j];
            }
        }

        return output;
    }

    vector<float> softMax(const vector<vector<float>>& input){
        int s = input[0].size();
        vector<float> output(s);
        float mother = 0;
        for(int j = 0; j < s; j++){
            mother += exp(input[0][j]);
        }
        for(int j = 0; j < s; j++){
            output[j] = exp(input[0][j]) / mother;
        }

        return output;
    }

    vector<vector<float>> init_matrix(const vector<int>& dims){
        return vector<vector<float>>(
            dims[0], 
            vector<float>(dims[1], 0.0f)
        );
    }

    void load_model(){
        ifstream file("../mnist-fc/meta.json");
        json data = json::parse(file);
        weight1 = init_matrix(data["fc1.weight"].get<vector<int>>());
        bias1 = init_matrix(data["fc1.bias"].get<vector<int>>());
        weight2 = init_matrix(data["fc2.weight"].get<vector<int>>());
        bias2 = init_matrix(data["fc2.bias"].get<vector<int>>());
    }

    void load_data(const string& bin_name, vector<vector<float>>& matrix){
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

        auto count = file_size / sizeof(float);

        vector<float> buffer(count);
        f.read(reinterpret_cast<char*>(buffer.data()), file_size);

        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                matrix[i][j] = buffer[i * col + j];
            }
        }
    }

public:
    
    Model(){
        load_model();
        load_data("../mnist-fc/fc1.weight", weight1);
        load_data("../mnist-fc/fc2.weight", weight2);
        load_data("../mnist-fc/fc1.bias", bias1);
        load_data("../mnist-fc/fc2.bias", bias2);
    }
    
    ~Model(){}

    vector<float> forward(const string& imagePath){
        vector<vector<float>> input = processImage(imagePath);
        vector<float> output;
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
            vector<float> imageData = forward(imagePath);
            
            vector<vector<float>> allImagesData;
    
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
