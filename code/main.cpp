// #include "ui.hpp"
#include "model.hpp"

int main(){
    cout << "type 1 if you choose mnist-fc model, type 2 if you choose mnist-fc-plus model" << endl;
    string choose;
    cin >> choose;

    auto model_ptr = create_model(choose);

    model_ptr->load_model(choose);
    model_ptr->process_all();
    



    //  //test
    //  vector<vector<float>> m1(1000,vector<float>(1000, 1.0f));
    //  vector<vector<float>> m2(1000,vector<float>(500, 2.0f));
     
    //  Model<float> model;
      
    //  auto start = std::chrono::high_resolution_clock::now(); 
    //  auto m3 = model.matrix_multiply(m1, m2); 
    //  auto end = std::chrono::high_resolution_clock::now();
    //  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    //  std::cout << "函数执行耗时: " << duration_ns.count() << " 纳秒" << std::endl;
    //  //test
    return 0;
}

