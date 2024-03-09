#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<double> softmax(vector<double> list)
{
    int size = list.size();
    vector<double> result(size);
    double sum = 0;

    for (int i = 0; i < size; i++)
    {
        result[i] = exp(list[i]);
        sum += result[i];
    }

    for (int i = 0; i < size; i++)
    {
        result[i] /= sum;
    }

    return result;
}

int main()
{
    // Test case 1: Positive numbers
    vector<double> input1 = {1.0, 2.0, 3.0};
    vector<double> output1 = softmax(input1);
    cout << "Test case 1: ";
    for (double val : output1)
    {
        cout << val << " ";
    }
    cout << endl;

    // Test case 2: Negative numbers
    vector<double> input2 = {-1.0, -2.0, -3.0};
    vector<double> output2 = softmax(input2);
    cout << "Test case 2: ";
    for (double val : output2)
    {
        cout << val << " ";
    }
    cout << endl;

    // Test case 3: Mixed positive and negative numbers
    vector<double> input3 = {-1.0, 2.0, -3.0};
    vector<double> output3 = softmax(input3);
    cout << "Test case 3: ";
    for (double val : output3)
    {
        cout << val << " ";
    }
    cout << endl;

    // Test case 4: Empty input
    vector<double> input4 = {};
    vector<double> output4 = softmax(input4);
    cout << "Test case 4: ";
    for (double val : output4)
    {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}