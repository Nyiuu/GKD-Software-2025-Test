#include <vector>
#include <cmath>
#include <iostream>
using namespace std;

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
        
        for(int i = 0; i < matrix1.size(); i++){
            for(int j = 0; j < matrix1[0].size(); j++){
                result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }

        return result;
    }

    vector<vector<float>> relu(const vector<vector<float>>& input){
        vector<vector<float>> output(input.size(), vector<float>(input[0].size()));
       
        for(int i = 0; i < input.size(); i++){
            for(int j = 0; j < input[0].size(); j++){
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

public:
    Model() : weight1(784, vector<float>(500)), bias1(1, vector<float>(500)),weight2(500, vector<float>(10)), bias2(1, vector<float>(10)){} 
    
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

int main(){
    Model model;
    vector<vector<float>> input(1, vector<float>(784, 1));
    auto output = model.forward(input);
    for(int i = 0; i < output.size(); i++){
        cout << output[i] << endl;
    }
}

