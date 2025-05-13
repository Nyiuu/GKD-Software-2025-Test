#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "modelbase.hpp"
#include <thread>
#include <mutex>

using json = nlohmann::json;
using namespace std;
using namespace cv;

template<typename T>
class Model : public ModelBase{
private:
    vector<vector<T>> weight1;
    vector<vector<T>> bias1;
    vector<vector<T>> weight2;
    vector<vector<T>> bias2;

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
            vector<T>(dims[1], T(0))
        );
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
    
        for(int i = 0; i < row; ++i){
            for(int j = 0; j < col; ++j){
                matrix[i][j] = buffer[i * col + j];
            }
        }
    }

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
        if constexpr(is_same_v<T, float>){
            flattened.convertTo(flattened, CV_32F); 
        }else if constexpr(is_same_v<T, double>){
            flattened.convertTo(flattened, CV_64F);  
        }
    
        // 4. 归一化到0-1范围
        vector<T> normalizedVec;
        flattened /= 255.0;
        normalizedVec.assign(flattened.begin<T>(), flattened.end<T>());
        vector<vector<T>> result;
        result.push_back(normalizedVec);
        return result;
    }

    


public:
    
    Model(){
        
    }
    
    ~Model(){}

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


    vector<vector<T>> parallel_matrix_multiply(const vector<vector<T>>& matrix1, const vector<vector<T>>& matrix2, unsigned int num_threads = 12){
        int row = matrix1.size();
        int col = matrix2[0].size();
        int w = matrix2.size();

        vector<vector<T>> result(row, vector<T>(col));

        int rows_per_thread = (row + num_threads - 1) / num_threads;

        vector<thread> threads;
        threads.reserve(num_threads);//reserve不创建具体对象

        //用lambda匿名函数便于直接使用函数内创建的变量
        auto worker = [&](int thread_id){
            int start_row = thread_id * rows_per_thread;
            // int end_row = (thread_id + 1) * rows_per_thread > row ? row : (thread_id + 1) * rows_per_thread;
            int end_row = min((thread_id + 1) * rows_per_thread, row);
            for(int i = start_row; i < end_row; i++){
                for(int j = 0; j < col; j++){
                    T sum = 0;
                    for(int k = 0; k < w; k++){
                        sum += matrix1[i][k] * matrix2[k][j];
                    }
                   result[i][j]= sum;
                }
            }
    
        };

        for(unsigned int i = 0; i < num_threads; i++){
            threads.emplace_back(worker, i);
        }

        for(auto& thread : threads){
            thread.join();
        }

        return result;

    }

    void load_model(const string& choose) override {
        string folder_path;
        if(choose == "1"){
            folder_path = "../mnist-fc/";
        }else if(choose == "2"){
            folder_path = "../mnist-fc-plus/";
        }
        ifstream file(folder_path + "meta.json");
        json data = json::parse(file);
        string type = data["type"];
        if(type == "fp32"){
            weight1 = init_matrix(data["fc1.weight"].get<vector<T>>());
            bias1 = init_matrix(data["fc1.bias"].get<vector<T>>());
            weight2 = init_matrix(data["fc2.weight"].get<vector<T>>());
            bias2 = init_matrix(data["fc2.bias"].get<vector<T>>());
            load_data(folder_path + "fc1.weight", weight1);
            load_data(folder_path + "fc2.weight", weight2);
            load_data(folder_path + "fc1.bias", bias1);
            load_data(folder_path + "fc2.bias", bias2);
        }else if(type == "fp64"){
            weight1 = init_matrix(data["fc1.weight"].get<vector<T>>());
            bias1 = init_matrix(data["fc1.bias"].get<vector<T>>());
            weight2 = init_matrix(data["fc2.weight"].get<vector<T>>());
            bias2 = init_matrix(data["fc2.bias"].get<vector<T>>());
            load_data(folder_path + "fc1.weight", weight1);
            load_data(folder_path + "fc2.weight", weight2);
            load_data(folder_path + "fc1.bias", bias1);
            load_data(folder_path + "fc2.bias", bias2);
        }else{
            return;
        }
         
    }

    vector<T> forward(const string& imagePath){
        vector<vector<T>> input = processImage(imagePath);
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

            // vector<T> imageData = forward(imagePath);
            
            //test
            auto start = std::chrono::high_resolution_clock::now();
            vector<T> imageData = forward(imagePath);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            std::cout << "函数执行耗时: " << duration_ns.count() << " 纳秒" << std::endl;
            //test

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

unique_ptr<ModelBase> create_model(const string& choose){
    if(choose == "1"){
        return make_unique<Model<float>>();
    }else if(choose == "2"){
        return make_unique<Model<double>>();
    }
}