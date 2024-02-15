#pragma once
#include <vector>

using std::vector;

class neuralnet
{
private:
    char *hidden_act_fun = "sig";

public:
    vector<double> hidden_act(vector<double> list);
};