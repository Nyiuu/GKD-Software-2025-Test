#include "model.hpp"
#include <iostream>
using namespace std;

int main(){
    Model model;
    vector<vector<float>> input(1, vector<float>(784, 1));
    auto output = model.forward(input);
    for(size_t i = 0; i < output.size(); i++){
        cout << output[i] << endl;
    }
}

