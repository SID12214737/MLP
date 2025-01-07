#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double logistic(double x) {
    return 1 / (1 + exp(-x));
}

double logistic_derivative(double x) {
    double s = logistic(x);
    return s * (1 - s);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Generate a random double values
double random_value() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Scale to [-1, 1]
}


// Softmax Function
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

vector<vector<double>> softmaxDerivative(const vector<double>& softmaxOutput) {
    size_t n = softmaxOutput.size();
    vector<vector<double>> jacobianMatrix(n, vector<double>(n));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                jacobianMatrix[i][j] = softmaxOutput[i] * (1 - softmaxOutput[i]);
            } else {
                jacobianMatrix[i][j] = -softmaxOutput[i] * softmaxOutput[j];
            }
        }
    }

    return jacobianMatrix;
}

// Loss function: Mean Squared Error
double mse_loss(const vector<double>& output, const vector<double>& target) {
    double loss = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        double diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / output.size();
}

// Loss function: Cross Entropy
double cross_entropy_loss(const vector<double>& output, const vector<double>& target) {
    double loss = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        loss -= target[i] * log(output[i] + 1e-15); // Add small value to avoid log(0)
    }
    return loss / output.size();
}

// Neuron
struct Neuron {
    vector<double> weights;
    double threshold = random_value();
    int activation_type = 0; // 0->Logistic, 1->ReLU, 2->Linear, 3->softmax

    Neuron(int activation, int n_ins) : activation_type(activation) {
        for (int i = 0; i < n_ins; i++) {
            weights.push_back(random_value());
        }
    }

    double transfer(const vector<double>& inputs) {
        double z = threshold;
        for (size_t i = 0; i < inputs.size(); ++i) {
            z += inputs[i] * weights[i];
        }
        return z;
    }

    double activate(const double x) {
        switch (activation_type) {
            case 0: return logistic(x);
            case 1: return relu(x);
            case 2: return x;
            default: return x;
        }
    }

    double activate_derivative(double x) {
        switch (activation_type) {
            case 0: return logistic_derivative(x);
            case 1: return relu_derivative(x);
            case 2: return 1;
            default: return 1;
        }
    }

    double forward(const vector<double>& inputs) {
        double output = transfer(inputs);
        output = activate(output);
        return output;
    }
};

// Layer
struct Layer {
    vector<Neuron> neurons;

    Layer(const int n_neurons,const int n_ins, const int activation = 0) {
        for (int i = 0; i < n_neurons; ++i) {
            neurons.emplace_back(activation, n_ins);
        }
    }
    
    vector<double> forward(const vector<double>& inputs) {
        vector<double> outputs;
        for (Neuron& neuron: neurons) {
            outputs.push_back(neuron.forward(inputs));
        }
        return outputs; 
    }
};

// Neural Network
struct NeuralNetwork {
    vector<Layer> layers;

    NeuralNetwork(const vector<int>& topology) {
        size_t shcema_size = topology.size();
        for (size_t i = 1; i < shcema_size; ++i) {
            layers.emplace_back(topology[i], topology[i - 1], 1);
        }
    }

    vector<double> forward(vector<double>& inputs) {
        vector<double> activations = inputs;
        for (Layer& layer: layers) {
            activations = layer.forward(activations);
        }
        return activations;
    }

    void backpropagate(const vector<double>& inputs, const vector<double>& target, double learning_rate) {
        vector<vector<double>> layer_outputs;
        vector<double> activations = inputs;

        // Forward pass: store outputs for each layer
        layer_outputs.push_back(activations);
        for (Layer& layer : layers) {
            activations = layer.forward(activations);
            layer_outputs.push_back(activations);
        }

        // Calculate output error
        vector<double> output_error(layer_outputs.back().size());
        for (size_t i = 0; i < output_error.size(); ++i) {
            output_error[i] = layer_outputs.back()[i] - target[i];
        }
        
        // Backward pass
        vector<vector<double>> deltas(layers.size());
        deltas.back() = output_error;
        for (int l = layers.size() - 1; l >= 0; --l) {
            Layer& layer = layers[l];
            deltas[l].resize(layer.neurons.size());

            for (size_t j = 0; j < layer.neurons.size(); ++j) {
                double delta = deltas[l][j] * layer.neurons[j].activate_derivative(layer_outputs[l + 1][j]);
                for (size_t i = 0; i < layer_outputs[l].size(); ++i) {
                    layer.neurons[j].weights[i] -= learning_rate * delta * layer_outputs[l][i];
                }
                layer.neurons[j].threshold -= learning_rate * delta;
            }

            // Calculate deltas for the next layer
            if (l > 0) {
                vector<double> next_deltas(layer_outputs[l].size());
                for (size_t i = 0; i < next_deltas.size(); ++i) {
                    double error = 0.0;
                    for (size_t j = 0; j < deltas[l].size(); ++j) {
                        error += deltas[l][j] * layers[l].neurons[j].weights[i];
                    }
                    next_deltas[i] = error;
                }
                deltas[l - 1] = next_deltas;
            }
        }
    }
};

int main() {
    srand(time(0)); // Seed random number generator

    vector<int> topology{2, 4, 2};
    NeuralNetwork nn(topology);

    vector<vector<double>> inputs{
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
    };
    vector<vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
    
    for (int epoch = 0; epoch < 1000; ++epoch) {
        double loss=0.0;
        for (size_t i = 0; i<inputs.size(); i++) {
            vector<double> output = nn.forward(inputs[i]);
            nn.backpropagate(inputs[i], targets[i], 0.1);
            loss = mse_loss(output, targets.back());
        }
        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << " Loss: " << loss << endl;
        }

    }
    cout << "\nTesting the XOR problem:" << endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        vector<double> output = nn.forward(inputs[i]);
        cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] "
             << "Output: " << output[0] << " "
             << "Expected: " << targets[i][0] << endl;
    }
    return 0;
}
