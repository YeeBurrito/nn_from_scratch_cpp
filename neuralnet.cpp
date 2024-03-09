#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <numeric>
#include "neuralnet.h"

using std::vector;

// Class constructor
neuralnet::neuralnet(int input_size, int nlayers, int output_size, char *hidden_act_fun, double init_range, int nunits, double learn_rate, int mb_size, char type)
{
    this->input_size = input_size;
    this->nlayers = nlayers;
    this->output_size = output_size;
    this->hidden_act_fun = hidden_act_fun;
    this->init_range = init_range;
    this->nunits = nunits;
    this->learn_rate = learn_rate;
    this->mb_size = mb_size;
    this->type = type;

    weights = new double**[nlayers + 1];
    biases = new double*[nlayers + 1];

    // Initialize weights and biases
    for (int i = 0; i < nlayers + 1; i++)
    {
        int wb_input_size = 0;
        int wb_output_size = 0;
        // Get the correct input and output size for the first and last layers
        if (i == 0)
        {
            wb_input_size = input_size;
        }
        else
        {
            wb_input_size = nunits;
        }
        if (i == nlayers)
        {
            wb_output_size = output_size;
        }
        else
        {
            wb_output_size = nunits;
        }
        
        weights[i] = new double*[wb_input_size];
        biases[i] = new double[wb_output_size];
  
        for (int j = 0; j < wb_input_size; j++)
        {
            weights[i][j] = new double[wb_output_size];

            for (int k = 0; k < wb_output_size; k++)
            {
                weights[i][j][k] = (double)rand() / RAND_MAX * init_range * 2 - init_range;
            }

            biases[i][j] = (double)rand() / RAND_MAX * init_range * 2 - init_range;
        }
    }
}

double* neuralnet::hidden_act(double* list)
{
    // Get the size of the list
    int size = sizeof(list) / sizeof(list[0]);
    double* result = new double[size];

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

double* neuralnet::deriv_hidden_act(double* list)
{
    int size = sizeof(list) / sizeof(list[0]);
    double* result = new double[size];

    if (strcmp(hidden_act_fun, "sig") == 0)
    {
        double* sig = hidden_act(list);
        for (int i = 0; i < size; i++)
        {
            result[i] = sig[i] * (1 - sig[i]);
        }
    }
    if (strcmp(hidden_act_fun, "tanh") == 0)
    {
        for (int i = 0; i < size; i++)
        {
            double* tanh = hidden_act(list);
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

double* neuralnet::softmax(double* list, int size)
{
    double* result = new double[size];
    double sum = 0;

    // Find the maximum value in the list
    double max_val = *std::max_element(list, list + size);

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

forward_pass_result neuralnet::forward(double* inputs)
{
    // Store the preactivations and activations for each layer
    double** preactivations = new double*[nlayers + 1];
    double** activations = new double*[nlayers + 1];

    double* layer_input = new double[input_size];

    for (int i = 0; i < input_size; i++)
    {
        layer_input[i] = inputs[i];
    }

    // Calculate the preactivations and activations for all of the hidden layers
    for (int i = 0; i < nlayers; i++)
    {
        double** layer_weights = weights[i];
        double* layer_biases = biases[i];

        double* layer_preactivation = new double[nunits];

        int layer_input_size = i == 0 ? input_size : nunits;

        for (int j = 0; j < nunits; j++)
        {
            double preactivation = 0;
            for (int k = 0; k < layer_input_size; k++)
            {
                preactivation += layer_weights[k][j] * layer_input[k];
            }
            preactivation += layer_biases[j];
            layer_preactivation[j] = preactivation;
        }

        preactivations[i] = layer_preactivation;

        double* layer_activation = hidden_act(layer_preactivation);

        // Update the layer input
        // This could be a different size than the input size
        layer_input = layer_activation;
    }

    // Calculate the preactivations and activations for the output layer
    double** output_weights = weights[nlayers];
    double* output_biases = biases[nlayers];

    double* output_preactivation(new double[output_size]);
    for (int i = 0; i < output_size; i++)
    {
        double preactivation = 0;
        for (int j = 0; j < nunits; j++)
        {
            preactivation += output_weights[j][i] * layer_input[j];
        }
        preactivation += output_biases[i];
        output_preactivation[i] = preactivation;
    }

    preactivations[nlayers] = output_preactivation;

    // If the problem is classification, use the softmax function
    double* output_activation;
    if (type == 'c')
    {
        output_activation = softmax(output_preactivation, output_size);
    }
    else
    {
        output_activation = output_preactivation;
    }

    activations[nlayers] = output_activation;

    forward_pass_result result;
    result.activations = activations;
    result.preactivations = preactivations;

    return result;
}

void neuralnet::backward(int* labels, forward_pass_result result, int num_classes)
{
    double** preactivations = result.preactivations; // Get the preactivations
    double** activations = result.activations;       // Get the activations

    double** one_hot_labels = new double*[num_classes]; // Initialize one_hot_labels
    double* error = new double[num_classes];           // Initialize the error

    // If it is a classification problem, use the one-hot encoder
    if (type == 'c')
    {
        one_hot_labels = one_hot_encoder(labels, num_classes);
        // Calculate the error
        for (int i = 0; i < error.size(); i++)
        {
            error[i] = activations[activations.size() - 1][i] - one_hot_labels[i][i];
        }
    }
    else
    {
        // Calculate the error
        for (int i = 0; i < error.size(); i++)
        {
            error[i] = activations[activations.size() - 1][i] - labels[i];
        }
    }

    // Calculate the gradients for the output layer
    vector<vector<vector<double>>> weights_deriv(nlayers + 1, vector<vector<double>>(nunits, vector<double>(nunits)));
    vector<double> biases_deriv(nunits);

    if (mb_size == 0)
    {
        mb_size = labels.size();
    }

    // Calculate the gradients for each layer and update them
    for (int i = nlayers; i >= 0; i--)
    {
        // Calculate the weights derivative
        // calculate the dot product of the error and the activations
        weights_deriv[i] = vector<vector<double>>(weights[i].size(), vector<double>(weights[i][0].size()));
        for (int j = 0; j < weights[i].size(); j++)
        {
            for (int k = 0; k < weights[i][j].size(); k++)
            {
                weights_deriv[i][j][k] = activations[i][j] * error[k];
                // multiply by 1 / mb_size
                weights_deriv[i][j][k] *= 1 / mb_size;
            }
        }
        // Calculate the biases derivative
        // This is the equivalent of biases_deriv[i] = (1/self.mb_size) * np.sum(error) in python
        biases_deriv = vector<double>(biases[i].size());
        for (int j = 0; j < biases[i].size(); j++)
        {
            biases_deriv[j] = error[j] * 1 / mb_size;
        }

        // Update the weights and biases
        for (int j = 0; j < weights[i].size(); j++)
        {
            for (int k = 0; k < weights[i][j].size(); k++)
            {
                weights[i][j][k] -= learn_rate * weights_deriv[i][j][k];
            }
        }
        for (int j = 0; j < biases[i].size(); j++)
        {
            biases[i][j] -= learn_rate * biases_deriv[j];
        }

        // Calculate the error for the next layer
        if (i > 0)
        {
            vector<double> next_error = vector<double>(nunits);
            for (int j = 0; j < nunits; j++)
            {
                next_error[j] = 0;
                for (int k = 0; k < nunits; k++)
                {
                    next_error[j] += weights[i][k][j] * error[k];
                }
            }
            error = next_error;
        }
    }
}

vector<vector<double>> one_hot_encoder(vector<int> labels, int num_classes)
{
    // Create arrays of zeros with length = num_classes

    vector<vector<double>> result(labels.size(), vector<double>(num_classes, 0));

    // make the index of the label 1, e.g. if label is 2, then the 2nd index of the array will be 1

    for (int i = 0; i < labels.size(); i++)
    {
        result[i][labels[i]] = 1;
    }

    result = transpose_2d_vector(result);

    return result;
}

vector<vector<double>> transpose_2d_vector(vector<vector<double>> v)
{
    int n = v.size();
    int m = v[0].size();
    vector<vector<double>> result(m, vector<double>(n));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            result[j][i] = v[i][j];
        }
    }

    return result;
}

int main()
{
    // test weight initialization
    // neuralnet(int input_size, int nlayers, int output_size, char *hidden_act_fun, double init_range, int nunits, double learn_rate, int mb_size, char type)
    neuralnet *nn = new neuralnet(5, 2, 2, (char*)"sig", 0.5, 3, 0.1, 32, 'c');
}