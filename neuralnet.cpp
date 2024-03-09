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
    }
    if (strcmp(hidden_act_fun, "tanh") == 0)
    {
        for (int i = 0; i < size; i++)
        {
            double *tanh = hidden_act(list, size);
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
}

void neuralnet::backward(double *labels, forward_pass_result result, int num_classes)
{
    double **preactivations = result.preactivations;
    double **activations = result.activations;
    int size = input_size;
    double **one_hot_labels = new double *[size]; // Initialize the one-hot labels
    double **error = new double *[size];          // Initialize the error

    // If it is a classification problem, use the one-hot encoder
    if (type == 'c')
    {
        one_hot_labels = one_hot_encoder(labels, num_labels, num_classes);

        // Calculate the error
        for (int i = 0; i < size; i++)
        {
            error[i] = new double[num_classes];
            for (int j = 0; j < num_classes; j++)
            {
                error[i][j] = activations[nlayers][j] - one_hot_labels[i][j];
            }
        }
    }
    else
    {
        // Calculate the error for a regression problem
        for (int i = 0; i < size; i++)
        {
            error[i] = new double[1];
            error[i][0] = activations[nlayers][0] - labels[i];
        }
    }

    if (mb_size == 0)
    {
        mb_size = size;
    }

    // Calculate the gradients for each layer and update them
    for (int i = nlayers; i >= 0; i--)
    {
        // Allocate memory for the weights and biases gradients
        double **weights_deriv = new double *[i == 0 ? input_size : nunits];
        double *biases_deriv = new double[i == nlayers ? output_size : nunits];

        // Weights_deriv = (1 / mb_size) * np.dot(error, activations[i].T)
        // Print the activations and error
        std::cout << "Activations: \n";
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < num_classes; j++)
            {
                std::cout << activations[nlayers][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Error: \n";
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < num_classes; j++)
            {
                std::cout << error[i][j] << " ";
            }
            std::cout << std::endl;
        }
        exit(1);
    }
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
    printf("Hello, World!\n");
    // test weight initialization
    // neuralnet(int input_size, int nlayers, int output_size, char *hidden_act_fun, double init_range, int nunits, double learn_rate, int mb_size, char type)
    neuralnet *nn = new neuralnet(5, 2, 1, (char *)"sig", 0.5, 3, 0.1, 0, 'r');
    // test forward pass
    double inputs[5] = {1, 2, 3, 4, 5};
    forward_pass_result result = nn->forward(inputs);
    // print the result
    std::cout << "Activations: \n";
    std::cout << result.activations[2][0] << std::endl;
    double labels[1] = {0.3};
    // print weights before
    std::cout << "Weights before: \n";
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cout << nn->weights[0][i][j] << " ";
        }
        std::cout << std::endl;
    }
    // test backward pass
    nn->backward(labels, 1, result, 1);
    // print weights after
    std::cout << "Weights after: \n";
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cout << nn->weights[0][i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}