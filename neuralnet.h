#pragma once

#include <vector>

struct forward_pass_result
{
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> preactivations;
};

class neuralnet
{
private:
    int input_size;                                        // Number of neurons in the input layer
    int nlayers;                                           // Number of hidden layers
    int output_size;                                       // Number of neurons in the output layer, also the number of classes
    char *hidden_act_fun;                                  // Hidden layer activation function
    double init_range;                                     // Range for the initial weights and biases
    int nunits;                                            // Number of neurons in each hidden layer
    double learn_rate;                                     // Learning rate
    int mb_size;                                           // Mini-batch size
    char type;                                             // Type of neural network, r for regression, c for classification
    std::vector<std::vector<std::vector<double>>> weights; // Weights for each layer
    std::vector<std::vector<double>> biases;               // Biases for each layer

public:
    // Class constructor
    neuralnet(int input_size, int nlayers, int output_size, char *hidden_act_fun, double init_range, int nunits, double learn_rate, int mb_size, char type);

    std::vector<double> hidden_act(std::vector<double> list);
    std::vector<double> deriv_hidden_act(std::vector<double> list);
    std::vector<double> softmax(std::vector<double> list);
    forward_pass_result forward(std::vector<double> inputs);
    void backward(std::vector<double> labels, forward_pass_result result, int num_classes);
};

std::vector<std::vector<double>> transpose_2d_vector(std::vector<std::vector<double>> v);
std::vector<std::vector<double>> one_hot_encoder(std::vector<double> labels, int num_classes);