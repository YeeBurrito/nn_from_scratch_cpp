#include <vector>

class neuralnet
{
private:
    char *hidden_act_fun = "sig";

public:
    std::vector<double> hidden_act(std::vector<double> list);
    std::vector<double> deriv_hidden_act(std::vector<double> list);
    std::vector<double> softmax(std::vector<double> list);
};