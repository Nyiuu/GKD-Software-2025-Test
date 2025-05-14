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

    bool read_matrix(stringstream& buffer_stream, vector<vector<T>>& matrix) {
        matrix.clear();
        string line;
        
        while(getline(buffer_stream, line) && !line.empty()){
            istringstream line_stream(line);
            vector<T> row;
            T value;
            
            while (line_stream >> value) {
                row.push_back(value);
            }
            
            if (!row.empty()) {
                matrix.push_back(row);
            }
        }
        
        bool has_data = !matrix.empty();

        string end_mark;
        if (has_data && buffer_stream >> end_mark) {
            buffer_stream.ignore();
        }
        return has_data;
    }

    vector<vector<vector<T>>> read_all(stringstream& buffer_stream) {
        vector<vector<vector<T>>> matrices;
        vector<vector<T>> matrix;
        
        // 循环读取所有矩阵，直到文件结束
        while (read_matrix(buffer_stream, matrix)) {
            matrices.push_back(matrix);
        }
        
        return matrices;
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


    vector<vector<T>> parallel_matrix_multiply(const vector<vector<T>>& matrix1, const vector<vector<T>>& matrix2, unsigned int num_threads = 5){
        int row = matrix1.size();
        int col = matrix2[0].size();
        int w = matrix2.size();

        vector<vector<T>> result(row, vector<T>(col));

        int cols_per_thread = (col + num_threads - 1) / num_threads;
       
        vector<thread> threads;
        threads.reserve(num_threads);//reserve不创建具体对象

        //用lambda匿名函数便于直接使用函数内创建的变量
        auto worker = [&](int thread_id){
            int start_col = thread_id * cols_per_thread;
            int end_col = std::min((thread_id + 1) * cols_per_thread, col);
            for(int j = start_col; j < end_col; j++){
                T sum = 0;
                for(int k = 0; k < w; k++){
                    sum += matrix1[0][k] * matrix2[k][j];
                }
                result[0][j] = sum;
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

    vector<T> forward(vector<vector<T>>& input){
        vector<T> output;
        auto temp = parallel_matrix_multiply(input, weight1, 5);
        temp = matrix_add(temp, bias1);
        temp = relu(temp);
        temp = parallel_matrix_multiply(temp, weight2);
        temp = matrix_add(temp, bias2);
        output = softMax(temp);
        return output;

        // vector<vector<T>> input = processImage(imagePath);
        // vector<T> output;
        // auto temp = matrix_multiply(input, weight1);
        // temp = matrix_add(temp, bias1);
        // temp = relu(temp);
        // temp = matrix_multiply(temp, weight2);
        // temp = matrix_add(temp, bias2);
        // output = softMax(temp);
        // return output;
    }


    void process_all(stringstream& buffer_stream) override{
        auto all_image_data = read_all(buffer_stream);
        for(size_t i = 0; i < all_image_data.size(); i++){
            auto image_data = forward(all_image_data[i]);
            cout << "\n第"+ to_string(i + 1) +"张图片的前10个归一化像素值:" << endl;
            for (size_t j = 0; j < 10 && j < image_data.size(); ++j) {
                cout << image_data[j] << " ";
            }
            cout << endl;
        }
    }
};

unique_ptr<ModelBase> create_model(const string& choose){
    if(choose == "1"){
        return make_unique<Model<float>>();
    }else if(choose == "2"){
        return make_unique<Model<double>>();
    }else{
        return nullptr;
    }
}