#pragma once
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>
#include <fstream>
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
            output[j] = input[0][j] / mother;
        }

        return output;
    }

    vector<vector<float>> init_matrix(const vector<int>& dims) {
        return vector<vector<float>>(
            dims[0], 
            vector<float>(dims[1], 0.0f)
        );
    }

    void load_model() {
        ifstream file("../mnist-fc/meta.json");
        json data = json::parse(file);
        weight1 = init_matrix(data["fc1.weight"].get<vector<int>>());
        bias1 = init_matrix(data["fc1.bias"].get<vector<int>>());
        weight2 = init_matrix(data["fc2.weight"].get<vector<int>>());
        bias2 = init_matrix(data["fc2.bias"].get<vector<int>>());
    }
public:
    
    Model(){
        load_model();
    }
    
    ~Model(){}

    vector<float> forward(const vector<vector<float>>& input){
        vector<float> output;
        auto temp = matrix_multiply(input, weight1);
        temp = matrix_add(temp, bias1);
        temp = relu(temp);
        temp = matrix_multiply(temp, weight2);
        temp = matrix_add(temp, bias2);
        output = softMax(temp);
        return output;
    }

    
};
