#include <iostream>
#include <vector>
#include <cstring>
#include "neuralnet.h"

using std::vector;

class neuralnet
{
private:
    char *hidden_act_fun = "sig";

public:
    vector<double> hidden_act(vector<double> list)
    {
        int size = list.size();
        vector<double> result(size);

        if (strcmp(hidden_act_fun, "sig") == 0)
        {
            for (int i = 0; i < size; i++)
            {
                result[i] = 1 / (1 + exp(-list[i]));
            }
        }
        else if (strcmp(hidden_act_fun, "tanh") == 0)
        {
            for (int i = 0; i < size; i++)
            {
                result[i] = tanh(list[i]);
            }
        }
        else if (strcmp(hidden_act_fun, "relu") == 0)
        {
            for (int i = 0; i < size; i++)
            {
                result[i] = list[i] > 0 ? list[i] : 0;
            }
        }
        else
        {
            std::cerr << "Error: Invalid hidden activation function\n";
            exit(1);
        }

        return result;
    }

    vector<double> deriv_hidden_act(vector<double> list)
    {
        int size = list.size();
        vector<double> result(size);

        if (strcmp(hidden_act_fun, "sig") == 0)
        {
            vector<double> sig = hidden_act(list);
            for (int i = 0; i < size; i++)
            {
                result[i] = sig[i] * (1 - sig[i]);
            }
        }
        // TODO: implement the rest of the activation functions
        else
        {
            std::cerr << "Error: Invalid hidden activation function\n";
            exit(1);
        }
    }
};

int main()
{
    return 0;
}