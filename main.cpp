#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Activation Functions
double logistic(double x) {
    return 1 / (1 + exp(-x));
}

double relu(double x) {
    return x > 0 ? x : 0;
}

// Generate a random double values
double random_value() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Scale to [-1, 1]
}


// Softmax Function (without std::max_element)
vector<double> softmax(const vector<double>& input) {
    // Step 1: Find the maximum value manually
    double max_val = input[0];
    for (double val : input) {
        if (val > max_val) max_val = val;
    }

    // Step 2: Compute the exponentials adjusted by max_val
    vector<double> exp_values(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        exp_values[i] = exp(input[i] - max_val);
    }

    // Step 3: Compute the sum of exponentials
    double sum_exp = 0.0;
    for (double val : exp_values) {
        sum_exp += val;
    }

    // Step 4: Normalize the exponentials
    for (double& val : exp_values) {
        val /= sum_exp;
    }

    return exp_values;
}

// Neuron
struct Neuron {
    vector<double> weights; // for the next layer
    double threshold = 0.0; // i still don't know how to use it so it's 0.0 for now
    int activation_type = 0; // 0->Logistic, 1->ReLU, 2->Linear

    Neuron(int activation = 0, int n_outs = 1) : activation_type(activation) {
        for (int i = 0; i < n_outs; i++) {
            weights.push_back(random_value());
        }
    }

    double transfer(const vector<double>& inputs) {
        double sum = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            sum += inputs[i];
        }
        return sum;
    }

    double activate(const vector<double>& inputs) {
        double neuron = transfer(inputs);

        switch (activation_type) {
            case 0: return logistic(neuron);
            case 1: return relu(neuron);
            case 2: return neuron;
            default: return neuron;
        }
    }

    vector<double> forward(const vector<double>& inputs) {
        vector<double> outputs;
        double active = activate(inputs);
        size_t weights_size = weights.size();
        
        for (int i = 0; i<weights_size; i++) {
            outputs.push_back(active * weights[i]);
        }

        return outputs;
    }
};

// Layer
struct Layer {
    vector<Neuron> neurons;

    Layer(const int num_neurons,const int n_outs, const int activation = 0) {
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(activation, n_outs);
        }

    }
    
    vector<vector<double>> forward(const vector<vector<double>>& inputs) {
        
        vector<vector<double>> outputs; // For next layer
        size_t neurons_size = neurons.size();

        for (size_t i = 0; i < neurons_size; i++) {
            outputs.push_back(neurons[i].forward(inputs[i]));    
        }

        return outputs; // These will serve as inputs for the next layer
    }
};

// Neural Network
struct NeuralNetwork {
    vector<Layer> layers; // Layers in the network

    NeuralNetwork(const vector<int>& schematics) {
        size_t shcema_size = schematics.size();
        for (size_t i = 0; i < shcema_size - 1; ++i) {
            layers.emplace_back(schematics[i], schematics[i + 1], 1);
        }
        layers.emplace_back(schematics.back(), schematics.back(), 2);
    }

    vector<vector<double>> forward(vector<vector<double>>& inputs) {
        for (auto& layer : layers) {
            inputs = layer.forward(inputs);
        }
        return inputs;
    }
};

int main() {
    srand(time(0)); // Seed random number generator

    // Define network with 2 inputs, 1 hidden layer (3 neurons), and 2 outputs
    vector<int> schematics{2, 3, 2};
    NeuralNetwork nn(schematics);

    // Example input
    vector<vector<double>> input = {{0.5}, {1.2}};
    vector<vector<double>> output = nn.forward(input);

    // Display output
    cout << "Network output: ";
    for (int i = 0; i < schematics.back(); i++){
        cout << output[i][0];
    }
    cout << endl;

    return 0;
}
