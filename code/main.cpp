// #include "ui.hpp"
#include "model.hpp"

int main(){
    cout << "type 1 if you choose mnist-fc model, type 2 if you choose mnist-fc-plus model" << endl;
    string choose;
    cin >> choose;

    auto model_ptr = create_model(choose);

    model_ptr->load_model(choose);
    model_ptr->process_all();
    

    return 0;
}

