#pragma once

#include <vector>

struct forward_pass_result
{
    double **activations;
    double **preactivations;
};

class neuralnet
{
private:
    int input_size;       // Number of neurons in the input layer
    int nlayers;          // Number of hidden layers
    int output_size;      // Number of neurons in the output layer, also the number of classes
    char *hidden_act_fun; // Hidden layer activation function
    double init_range;    // Range for the initial weights and biases
    int nunits;           // Number of neurons in each hidden layer
    double learn_rate;    // Learning rate
    int mb_size;          // Mini-batch size
    char type;

    // Type of neural network, r for regression, c for classification
    // std::vector< std::vector< std::vector<double> > >* weights; // Weights for each layer
    // std::vector< std::vector<double> >* biases;               // Biases for each layer

public:
    double ***weights;
    double **biases;

    // Class constructor
    neuralnet(int input_size, int nlayers, int output_size, char *hidden_act_fun, double init_range, int nunits, double learn_rate, int mb_size, char type);

    double *hidden_act(double *list, int size);
    double *deriv_hidden_act(double *list, int size);
    double *softmax(double *list, int size);
    forward_pass_result *forward(double **inputs, int num_inputs);
    void backward(double *labels, int num_labels, forward_pass_result *result, int num_classes);
};

double **one_hot_encoder(double *labels, int num_labels, int num_classes);
double dot_product(double *list1, double *list2, int size);
double **transpose_2d_vector(double **v, int n, int m);