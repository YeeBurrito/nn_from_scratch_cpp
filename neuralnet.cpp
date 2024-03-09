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

    this->weights = new double**[nlayers + 1];
    this->biases = new double*[nlayers + 1];

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
                weights[i][j][k] = (double)rand() / RAND_MAX * init_range;
            }
        }

        for (int j = 0; j < wb_output_size; j++)
        {
            biases[i][j] = (double)rand() / RAND_MAX * init_range;
        }
    }
}

double* neuralnet::hidden_act(double* list, int size)
{
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

double* neuralnet::deriv_hidden_act(double* list, int size)
{
    double* result = new double[size];

    if (strcmp(hidden_act_fun, "sig") == 0)
    {
        double* sig = hidden_act(list, size);
        for (int i = 0; i < size; i++)
        {
            result[i] = sig[i] * (1 - sig[i]);
        }
    }
    if (strcmp(hidden_act_fun, "tanh") == 0)
    {
        for (int i = 0; i < size; i++)
        {
            double* tanh = hidden_act(list, size);
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

        double* layer_activation = hidden_act(layer_preactivation, nunits);

        // Update the layer input
        // This could be a different size than the input size
        layer_input = layer_activation;
    }

    // Calculate the preactivations and activations for the output layer
    double** output_weights = weights[nlayers];
    double* output_biases = biases[nlayers];

    double* output_preactivation = new double[output_size];
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

void neuralnet::backward(double* labels, int num_labels, forward_pass_result result, int num_classes) {
    double** preactivations = result.preactivations;
    double** activations = result.activations;
    int size = num_labels;
    double** one_hot_labels = new double*[size]; // Initialize the one-hot labels
    double** error = new double*[size]; // Initialize the error

    // If it is a classification problem, use the one-hot encoder
    if (type == 'c') {
        one_hot_labels = one_hot_encoder(labels, num_labels, num_classes);

        // Calculate the error
        for (int i = 0; i < size; i++) {
            error[i] = new double[num_classes];
            for (int j = 0; j < num_classes; j++) {
                error[i][j] = activations[nlayers][j] - one_hot_labels[i][j];
            }
        }
    } else {
        // Calculate the error for a regression problem
        for (int i = 0; i < size; i++) {
            error[i] = new double[1];
            error[i][0] = activations[nlayers][0] - labels[i];
        }
    }

    if (mb_size == 0) {
        mb_size = size;
    }

    // Calculate the gradients for each layer and update them
    for (int i = nlayers; i >= 0; i--) {
        // Allocate memory for the weights and biases gradients
        double** weights_gradient = new double*[i == 0 ? input_size : nunits];
        double* biases_gradient = new double[i == nlayers ? output_size : nunits];

        // Calculate the gradients for the weights by doing the dot product of the error and the activations
        for (int j = 0; j < (i == 0 ? input_size : nunits); j++) {
            weights_gradient[j] = new double[i == nlayers ? output_size : nunits];
            for (int k = 0; k < (i == nlayers ? output_size : nunits); k++) {
                double sum = 0;
                for (int l = 0; l < size; l++) {
                    sum += error[l][k] * (i == 0 ? activations[i][j] : activations[i - 1][j]);
                }
                weights_gradient[j][k] = sum / mb_size;
            }
        }

        // Calculate the gradients for the biases
        for (int j = 0; j < (i == nlayers ? output_size : nunits); j++) {
            double sum = 0;
            for (int k = 0; k < size; k++) {
                sum += error[k][j];
            }
            biases_gradient[j] = sum / mb_size;
        }

        // Update the weights and biases
        for (int j = 0; j < (i == 0 ? input_size : nunits); j++) {
            for (int k = 0; k < (i == nlayers ? output_size : nunits); k++) {
                weights[i][j][k] -= learn_rate * weights_gradient[j][k];
            }
            biases[i][j] -= learn_rate * biases_gradient[j];
        }

        // Update the error
        // Equivalent of error = np.dot(self.weights[i].T, error) * self.deriv_hidden_act(preactivations[i])
        double** new_error = new double*[size];
        if (i != 0) {
            for (int j = 0; j < size; j++) {
                new_error[j] = new double[i == 0 ? input_size : nunits];
                for (int k = 0; k < (i == 0 ? input_size : nunits); k++) {
                    double sum = 0;
                    for (int l = 0; l < (i == nlayers ? output_size : nunits); l++) {
                        sum += weights[i][k][l] * error[j][l];
                    }
                    new_error[j][k] = sum * deriv_hidden_act(preactivations[i - 1], i == 0 ? input_size : nunits)[k];
                }
            }
        }

        // Update the error
        for (int j = 0; j < size; j++) {
            delete[] error[j];
        }
        delete[] error;
        error = new_error;
    }

    // Clean up memory
    for (int i = 0; i < size; i++) {
        delete[] one_hot_labels[i];
    }
    delete[] one_hot_labels;
}

// One-hot encoder
// This function takes a list of labels and the number of classes and returns a one-hot encoded list
// Example usage: one_hot_encoder([1, 2, 3], 4)
// Output: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
double** one_hot_encoder(double* labels, int num_labels, int num_classes)
{
    // Create arrays of zeros with length = num_classes
    int size = num_labels;
    double** result = new double*[size];

    // Convert the labels from double to int
    int* int_labels = new int[size];
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

double** transpose_2d_vector(double** v, int n, int m)
{
    double** result = new double*[m];

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
    neuralnet *nn = new neuralnet(5, 2, 1, (char*)"sig", 0.5, 3, 0.1, 0, 'r');
    // test forward pass
    double inputs[5] = {1, 2, 3, 4, 5};
    forward_pass_result result = nn->forward(inputs);
    // print the result
    std::cout << "Activations: \n";
    std::cout << result.activations[2][0] << std::endl;
    double labels[1] = {0.3};
    // print weights before
    std::cout << "Weights before: \n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << nn->weights[0][i][j] << " ";
        }
        std::cout << std::endl;
    }
    //test backward pass
    nn->backward(labels, 1, result, 1);
    // print weights after
    std::cout << "Weights after: \n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << nn->weights[0][i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}