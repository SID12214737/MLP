#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double logistic(double x){
    return 1 / (1 + exp(-x));
}

double relu(double x){
    return x > 0 ? x: 0;
}

double random_value(){
    return 1 + ((double)rand() / RAND_MAX) * 2;
}

struct Neuron{
    vector<double> weights;
    double threshold = 0.0;
    int activation = 0; // 0->Sigmoid, 1->ReLU
    
    Neuron(int activation = 0, int n_neurons){
        activation = activation;
        for (int i = 0; i < n_neurons; i++){
            weights.push_back(random_value());    
        }
    }

    double transfer(vector<double> inputs){
        double sum = 0.0;
        for (int i=0; i<inputs.size(); i++){
            sum += inputs[i];
        }
        return sum;
    }

    double activate(vector<double> inputs){
        double neuron = transfer(inputs);
        if (activation){
            return logistic(neuron);
        } else {
            return relu(neuron);
        }
    }
};

struct Layer{
    vector<Neuron> neurons;

    Layer(int NoN, int n_neurons, int activaton = 0){
        for (int i = 0; i < NoN; i++){
            neurons.push_back(Neuron(activaton=activaton, n_neurons=n_neurons));
        }
    }
    
};

struct NeuralNetwork{
    vector<Layer> hidden_layers;

    NeuralNetwork(vector<int> schematics){
        for (int i = 1; i < schematics.size()-1; i++){
            
        }
    }
};

int main(){
    vector<int> schematics{2, 3, 2};

    return 0;
}