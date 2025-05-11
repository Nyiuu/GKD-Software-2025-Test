#pragma once
#include <string>
#include <vector>
using namespace std;

class ModelBase{
public:
    virtual void load_model(const string& path) = 0;
    virtual void process_all() = 0;
    virtual ~ModelBase() = default;

};