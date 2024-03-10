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

    this->weights = new double **[nlayers + 1];
    this->biases = new double *[nlayers + 1];

    srand(time(NULL));

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

        weights[i] = new double *[wb_input_size];
        biases[i] = new double[wb_output_size];

        for (int j = 0; j < wb_input_size; j++)
        {
            weights[i][j] = new double[wb_output_size];
            for (int k = 0; k < wb_output_size; k++)
            {
                weights[i][j][k] = (double)rand() / RAND_MAX * init_range;
            }
        }

        for (int j = 0; j < wb_output_size; j++)
        {
            biases[i][j] = (double)rand() / RAND_MAX * init_range;
        }
    }
}

double *neuralnet::hidden_act(double *list, int size)
{
    double *result = new double[size];

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

double *neuralnet::deriv_hidden_act(double *list, int size)
{
    double *result = new double[size];

    if (strcmp(hidden_act_fun, "sig") == 0)
    {
        double *sig = hidden_act(list, size);
        for (int i = 0; i < size; i++)
        {
            result[i] = sig[i] * (1 - sig[i]);
        }
        delete[] sig;
    }
    else if (strcmp(hidden_act_fun, "tanh") == 0)
    {
        double *tanh = hidden_act(list, size);
        for (int i = 0; i < size; i++)
        {
            result[i] = 1 - tanh[i] * tanh[i];
        }
        delete[] tanh;
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

double *neuralnet::softmax(double *list, int size)
{
    double *result = new double[size];
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

// Forward pass
// This function takes the inputs and calculates the preactivations and activations for each layer,
// then returns the preactivations and activations for each layer. This works on multiple inputs at once.
forward_pass_result *neuralnet::forward(double **inputs, int num_inputs)
{
    // Create a list for each input, which will store the activations and preactivations for each layer
    forward_pass_result *results = new forward_pass_result[num_inputs];

    // Calculate a forward pass for each input
    for (int i = 0; i < num_inputs; i++)
    {
        // Get the current input
        double *current_input = inputs[i];

        // Create a list for each layer, which will store the preactivations and activations for each layer
        forward_pass_result result;
        result.preactivations = new double *[nlayers + 1];
        result.activations = new double *[nlayers + 1];

        // Calculate the preactivations and activations for all of the hidden layers
        for (int layer_index = 0; layer_index < nlayers; layer_index++)
        {
            // Get the weights and biases for the current layer
            double **current_weights = weights[layer_index];
            double *current_biases = biases[layer_index];

            // Calculate the preactivations for the current layer
            double *preactivations = new double[nunits];
            for (int j = 0; j < nunits; j++)
            {
                preactivations[j] = 0;
                for (int k = 0; k < nunits; k++)
                {
                    preactivations[j] += current_weights[k][j] * current_input[k];
                }
                preactivations[j] += current_biases[j];
            }

            // Calculate the activations for the current layer
            double *activations = hidden_act(preactivations, nunits);

            // Store the preactivations and activations for the current layer
            result.preactivations[layer_index] = preactivations;
            result.activations[layer_index] = activations;

            // Set the current input to the activations for the next layer
            current_input = activations;
        }

        // Calculate the preactivations and activations for the output layer
        double **output_weights = weights[nlayers];
        double *output_biases = biases[nlayers];

        // Calculate the preactivations for the output layer
        double *output_preactivations = new double[output_size];
        for (int j = 0; j < output_size; j++)
        {
            output_preactivations[j] = 0;
            for (int k = 0; k < nunits; k++)
            {
                output_preactivations[j] += output_weights[k][j] * current_input[k];
            }
            output_preactivations[j] += output_biases[j];
        }

        result.preactivations[nlayers] = output_preactivations;

        // Calculate the activations for the output layer
        double *output_activations;
        // If it is a classification problem, use the softmax function
        if (type == 'c')
        {
            output_activations = softmax(output_preactivations, output_size);
        }
        else
        {
            output_activations = output_preactivations;
        }

        result.activations[nlayers] = output_activations;

        // Store the results for the current input
        results[i] = result;
    }

    return results;
}

// Backward pass
// This function takes the labels and the results of the forward pass and calculates the gradients for each layer
// and updates the weights and biases
void neuralnet::backward(double *labels, int num_labels, forward_pass_result *result, int num_classes)
{
    // First, find the error for the output layer on each of the labels
    double **error = new double *[num_labels];

    // If it is a classification problem, use the one-hot encoder
    if (type == 'c')
    {
        double **one_hot_labels = one_hot_encoder(labels, num_labels, num_classes);
        for (int i = 0; i < num_labels; i++)
        {
            error[i] = new double[num_classes];
            for (int j = 0; j < num_classes; j++)
            {
                error[i][j] = result[i].activations[nlayers][j] - one_hot_labels[i][j];
            }
        }
    }
    else
    {
        for (int i = 0; i < num_labels; i++)
        {
            error[i] = new double[num_classes];
            for (int j = 0; j < num_classes; j++)
            {
                error[i][j] = result[i].activations[nlayers][j] - labels[i];
            }
        }
    }

    if (mb_size == 0)
    {
        mb_size = num_labels;
    }

    // Calculate the gradients for each label
    for (int i = 0; i < num_labels; i++)
    {
        // Calculate the gradients for each layer and update the weights and biases
        for (int layer_index = nlayers; layer_index >= 0; layer_index--)
        {
            // Get the activations for the current layer
            double *current_activations = result[i].activations[layer_index];
            double *current_preactivations = result[i].preactivations[layer_index];

            // Calculate the gradients for the output layer
            if (layer_index == nlayers)
            {
                double weights_gradient = dot_product(current_activations, error[i], num_classes);
                double biases_gradient = std::accumulate(error[i], error[i] + num_classes, 0.0);

                weights_gradient /= mb_size;
                biases_gradient /= mb_size;

                // Update the weights and biases for the output layer
                for (int j = 0; j < nunits; j++)
                {
                    for (int k = 0; k < num_classes; k++)
                    {
                        weights[nlayers][j][k] -= learn_rate * weights_gradient;
                    }
                    biases[nlayers][j] -= learn_rate * biases_gradient;
                }

                // Calculate the error for the next layer
                double *next_error = new double[nunits];
                double dot = dot_product(weights[nlayers][0], error[i], num_classes);
                double *deriv = deriv_hidden_act(current_preactivations, nunits);
                for (int j = 0; j < nunits; j++)
                {
                    next_error[j] = dot * deriv[j];
                }

                // Update the error for the next layer
                delete[] error[i];
                error[i] = next_error;
            }

            // Calculate the gradients for the hidden layers
            else
            {
                double **next_weights = weights[layer_index + 1];
                double *next_error = error[i];
                double *deriv = deriv_hidden_act(current_preactivations, nunits);

                // Calculate the gradients for the weights and biases
                for (int j = 0; j < nunits; j++)
                {
                    for (int k = 0; k < num_classes; k++)
                    {
                        double weights_gradient = current_activations[j] * next_error[k];
                        weights_gradient /= mb_size;
                        weights[layer_index][j][k] -= learn_rate * weights_gradient;
                    }
                    double biases_gradient = next_error[j];
                    biases_gradient /= mb_size;
                    biases[layer_index][j] -= learn_rate * biases_gradient;
                }

                // Calculate the error for the next layer
                if (layer_index > 0)
                {
                    double *next_error = new double[nunits];
                    for (int j = 0; j < nunits; j++)
                    {
                        next_error[j] = dot_product(next_weights[j], next_error, num_classes) * deriv[j];
                    }

                    // Update the error for the next layer
                    delete[] error[i];
                    error[i] = next_error;
                }
            }
        }
    }

    // Cleanup
    for (int i = 0; i < num_labels; i++)
    {
        delete[] error[i];
    }
    delete[] error;
}

// One-hot encoder
// This function takes a list of labels and the number of classes and returns a one-hot encoded list
// Example usage: one_hot_encoder([1, 2, 3], 4)
// Output: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
double **one_hot_encoder(double *labels, int num_labels, int num_classes)
{
    // Create arrays of zeros with length = num_classes
    int size = num_labels;
    double **result = new double *[size];

    // Convert the labels from double to int
    int *int_labels = new int[size];
    for (int i = 0; i < size; i++)
    {
        int_labels[i] = (int)labels[i];
    }

    // make the index of the label 1, e.g. if label is 2, then the 2nd index of the array will be 1
    for (int i = 0; i < size; i++)
    {
        result[i] = new double[num_classes];
        for (int j = 0; j < num_classes; j++)
        {
            result[i][j] = j == labels[i] ? 1 : 0;
        }
    }

    return result;
}

// Dot product
// This function takes two lists and returns the dot product of the two lists
// Example usage: dot_product([1, 2, 3], [4, 5, 6])
// Output: 32
double dot_product(double *list1, double *list2, int size)
{
    double result = 0;

    for (int i = 0; i < size; i++)
    {
        result += list1[i] * list2[i];
    }

    return result;
}

double **transpose_2d_vector(double **v, int n, int m)
{
    double **result = new double *[m];

    for (int i = 0; i < m; i++)
    {
        result[i] = new double[n];
        for (int j = 0; j < n; j++)
        {
            result[i][j] = v[j][i];
        }
    }

    return result;
}

int main()
{
    // neuralnet(int input_size, int nlayers, int output_size, char *hidden_act_fun, double init_range, int nunits, double learn_rate, int mb_size, char type)
    neuralnet *nn = new neuralnet(5, 2, 1, (char *)"sig", 1.0, 2, 0.1, 0, 'r');
}