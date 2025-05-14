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
class Matrix{
private:
    vector<vector<T>> data;
    size_t rows;
    size_t cols;
public:
    Matrix() : rows(0), cols(0){}
    Matrix(size_t r, size_t c, T v = T()) : data(r, vector<T>(c, v)), rows(r), cols(c){}

    size_t get_rows() const{
        return rows;
    }

    size_t get_cols() const{
        return cols;
    }

    void clear(){
        data.clear();
        rows = 0;
        cols = 0;
        data.resize(0);
    }
   
    vector<T>& operator[](size_t i){
        return data[i];
    }
    const vector<T>& operator[](size_t i) const {
        return data[i];
    }

    Matrix<T> operator+(const Matrix& other) const {
        // if (rows != other.rows || cols != other.cols) {
        //     throw invalid_argument("不能加");
        // }
        Matrix result(rows, cols);
        for(size_t i = 0; i < rows; ++i){
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = data[i][j] + other[i][j];
            }
        }
        return result;
    }

    Matrix<T> operator*(const Matrix& other) const {
        size_t num_threads = 5;
        size_t row = rows;
        size_t col = other.cols;
        size_t w = other.rows;
        
        // if (cols != w) {
        //     throw invalid_argument("不能乘");
        // }

        Matrix result(row, col);

        size_t cols_per_thread = (col + num_threads - 1) / num_threads;
        
        vector<thread> threads;
        threads.reserve(num_threads);

        auto worker = [&](size_t thread_id){
            size_t start_col = thread_id * cols_per_thread;
            size_t end_col = min((thread_id + 1) * cols_per_thread, col);
            for(size_t j = start_col; j < end_col; j++){              
                for(size_t i = 0; i < row; i++){
                    for(size_t k = 0; k < w; k++){
                        result[i][j] += data[i][k] * other[k][j];
                    }
                }      
            }
        };
    

        for(size_t i = 0; i < num_threads; i++){
            threads.emplace_back(worker, i);
        }

        for(auto& thread : threads){
            thread.join();
        }

        return result;


    }
    Matrix<T> relu() const {
        Matrix result(rows, cols);
        for(size_t i = 0; i < rows; ++i){
            for(size_t j = 0; j < cols; ++j){
                result[i][j] = data[i][j] < 0 ? T(0) : data[i][j];
            }
        }
        return result;
    }
    
    vector<T> softmax() const {
        // if(rows != 1){
        //     throw invalid_argument("只针对行向量");
        // }
        
        vector<T> output(cols);
        T mother = 0;
        for(size_t j = 0; j < cols; j++){
            mother += exp(data[0][j]);
        }
        for(size_t j = 0; j < cols; j++){
            output[j] = exp(data[0][j]) / mother;
        }

        return output;
        
    }

    void init_matrix(const vector<T>& dims){
       rows = static_cast<size_t>(dims[0]);
       cols = static_cast<size_t>(dims[1]);    
       data.resize(rows, vector<T>(cols, T(0)));
    }

    void load_data(const string& bin_name){    
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
    
        for(size_t i = 0; i < rows; ++i){
            for(size_t j = 0; j < cols; ++j){
                data[i][j] = buffer[i * cols + j];
            }
        }
    }

    bool read_matrix(stringstream& buffer_stream) {
        clear();
        string line;
        size_t maxCols = 0; 
        
        while(getline(buffer_stream, line)){
            if(line == "E"){
                break;
            }

            if(line.empty()){
                continue;
            }

            istringstream line_stream(line);
            vector<T> row;
            T value;    
            
            while (line_stream >> value) {
                row.push_back(value);
            }
            
            if (!row.empty()) {
                data.push_back(row);
                maxCols = max(maxCols, row.size()); 
            }
        }

        rows = data.size();
        cols = maxCols;

        return !data.empty();
    }

    
};



template<typename T>
class Model : public ModelBase{
private:
    Matrix<T> weight1;
    Matrix<T> bias1;
    Matrix<T> weight2;
    Matrix<T> bias2;

public:
    
    Model(){
        
    }
    
    ~Model(){}

    vector<Matrix<T>> read_all(stringstream& buffer_stream) {
        vector<Matrix<T>> matrices;
        Matrix<T> matrix;
        
        while(matrix.read_matrix(buffer_stream)){
            matrices.push_back(matrix);
        }
        
        return matrices;
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
            weight1.init_matrix(data["fc1.weight"].get<vector<T>>());
            bias1.init_matrix(data["fc1.bias"].get<vector<T>>());
            weight2.init_matrix(data["fc2.weight"].get<vector<T>>());
            bias2.init_matrix(data["fc2.bias"].get<vector<T>>());
            weight1.load_data(folder_path + "fc1.weight");
            weight2.load_data(folder_path + "fc2.weight");
            bias1.load_data(folder_path + "fc1.bias");
            bias2.load_data(folder_path + "fc2.bias");
        }else if(type == "fp64"){
            weight1.init_matrix(data["fc1.weight"].get<vector<T>>());
            bias1.init_matrix(data["fc1.bias"].get<vector<T>>());
            weight2.init_matrix(data["fc2.weight"].get<vector<T>>());
            bias2.init_matrix(data["fc2.bias"].get<vector<T>>());
            weight1.load_data(folder_path + "fc1.weight");
            weight2.load_data(folder_path + "fc2.weight");
            bias1.load_data(folder_path + "fc1.bias");
            bias2.load_data(folder_path + "fc2.bias");
        }else{
            return;
        }
    }

    vector<T> forward(Matrix<T>& input){
        auto temp = input * weight1;
        temp = temp.relu();
        temp = temp * weight2;
        temp = temp + bias2;
        auto output = temp.softmax();
        return output;
    }


    string process_all(stringstream& buffer_stream) override{
        string result;
        auto all_image_data = read_all(buffer_stream);
        for(size_t i = 0; i < all_image_data.size(); i++){
            auto image_data = forward(all_image_data[i]);
            result += "\n第" + to_string(i + 1) + "张图片的前10个归一化像素值:\n";
            for (size_t j = 0; j < 10 && j < image_data.size(); ++j) {
                result += to_string(image_data[j]) + " ";
            }
            result += "\n";
        }
        return result;
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