#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include "neuralnet.h"

using std::vector;

class neuralnetclass : public neuralnet
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
        if (strcmp(hidden_act_fun, "tanh") == 0)
        {
            for (int i = 0; i < size; i++)
            {
                vector<double> tanh = hidden_act(list);
                result[i] = 1 - tanh[i] * tanh[i];
            }
        }
        else if (strcmp(hidden_act_fun, "relu") == 0)
        {
            for (int i = 0; i < size; i++)
            {
                result[i] = list[i] >= 0 ? 1 : 0;
            }
        }
        else
        {
            std::cerr << "Error: Invalid hidden activation function\n";
            exit(1);
        }

        return result;
    }

    vector<double> softmax(vector<double> list)
    {
        int size = list.size();
        vector<double> result(size);
        double sum = 0;

        // Find the maximum value in the list
        double max_val = *max_element(list.begin(), list.end());

        for (int i = 0; i < size; i++)
        {
            // Subtract the maximum value before exponentiating
            result[i] = exp(list[i] - max_val);
            sum += result[i];
        }

        for (int i = 0; i < size; i++)
        {
            result[i] /= sum;
        }

        return result;
    }
};

int main()
{
    // test softmax
    vector<double> list = {3.2, 1.3, 0.2, 747.0};

    neuralnetclass nn;
    vector<double> result = nn.softmax(list);

    for (int i = 0; i < result.size(); i++)
    {
        std::cout << result[i] << " ";
    }
}