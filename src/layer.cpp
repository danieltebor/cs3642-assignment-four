#include <math.h>
#include <stdexcept>

#include "layer.hpp"

// The constructor takes the number of number of neurons, the number of connections that each
// neuron has to the next layer, and the activation function for the layer outputs.
// The weights vector is initialized to the size of the number of neurons in the current layer
// times the number of neurons in the next layer. The biases vector is initialized to the number
// of connections per neuron to the next layer. The weights and biases are initialized to random
// values between -1 and 1.
Layer::Layer(unsigned int num_neurons, unsigned int num_connections_per_neuron)
    : _weights(num_neurons * num_connections_per_neuron), _biases(num_connections_per_neuron) {
    // Reject a layer with 0 neurons or 0 connections per neuron.
    if (num_neurons == 0 || num_connections_per_neuron == 0) {
        throw std::invalid_argument("The number of neurons and connections per neuron must be greater than 0.");
    }

    this->_num_neurons = num_neurons;
    this->_num_connections_per_neuron = num_connections_per_neuron;

    // Initialize the weights to random values between -1 and 1.
    for (unsigned int i = 0; i < _weights.size(); i++) {
        _weights[i] = (float)std::rand() / RAND_MAX * 2 - 1;
    }

    // Initialize the biases to random values between -1 and 1.
    for (unsigned int i = 0; i < _biases.size(); i++) {
        _biases[i] = (float)std::rand() / RAND_MAX * 2 - 1;
    }
}

// Forward pass through the layer.
// The input vector is the input to the layer and must be the same size as the 
// layer input size. Returns the output of the layer.
std::vector<float> Layer::_forward(const std::vector<float>& input) {
    // The output of the layer.
    std::vector<float> output;
    output.reserve(_num_connections_per_neuron);
    
    // Calculate each output of the layer.
    // #pragma omp parallel for allows for parallelization of the loop using OpenMP, which is a library for
    // parallel programming in C++ that is enabled through the -fopenmp compiler flag.
    #pragma omp parallel for
    for (std::size_t connection_idx = 0; connection_idx < _num_connections_per_neuron; connection_idx++) {
        // The output of the current neuron.
        float current_output_value = 0.0f;

        for (std::size_t neuron_idx = 0; neuron_idx < _num_neurons; neuron_idx++) {
            // Add current layer_input * the weight for the current connection.
            current_output_value += input[connection_idx] * _weights[connection_idx * _num_neurons + neuron_idx];
        }

        // Add the bias for the current neuron.
        current_output_value += _biases[connection_idx];

        // Add the output of the current neuron to the output of the current layer.
        output.push_back(current_output_value);
    }

    // Store the output of the layer for the backward pass.
    _current_output = output;
    return output;
}

// Backward pass through the network.
// The prev_gradient vector is the gradient calculated by the loss function.
// The gradient is calculated by multiplying the sum of the gradients in the previous
// layer by multiplied by the weights they are attached to for each neuron.
// The bias_gradient vector is the gradient of the biases of the layer which is the sum
// of the gradients in the previous layer for each value in the bias_gradient.
// This applies the chain rule of calculus.
std::vector<float> Layer::backward(const std::vector<float>& prev_gradient) {
    std::vector<float> output_gradient(_num_neurons);
    std::vector<float> bias_gradient(_num_connections_per_neuron);

    // Calculate the gradient of each neuron in the layer.
    // The gradient of each neuron is the sum of the gradient of each connection to the next layer
    // multiplied by the weight of the connection.
    // #pragma omp parallel for allows for parallelization of the loop using OpenMP, which is a library for
    // parallel programming in C++ that is enabled through the -fopenmp compiler flag.
    #pragma omp parallel for
    for (std::size_t i = 0; i < _num_neurons; i++) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < _num_connections_per_neuron; j++) {
            sum += prev_gradient[j] * _weights[i * _num_connections_per_neuron + j];
            bias_gradient[j] += prev_gradient[j];
        }
        output_gradient[i] = sum * _derivative(_current_output[i]);
    }

    // Store the gradients of the layer for the optimizer.
    _current_weight_gradient = output_gradient;
    _current_bias_gradient = bias_gradient;
    return output_gradient;
}

// The derivative of the activation function is used to calculate the gradient of the layer
// during the backward pass.

// Calling the class as a function executes a forward pass through the layer.

// Linear activation function has a derivative of 1 since it does not use an activation function on the output.
inline float Linear::_derivative(float value) const {
    return 1.0f;
}

// Linear activation function does not use an activation function on the output.
std::vector<float> Linear::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    auto output = _forward(input);

    return output;
}

// Sigmoid activation function has a derivative of sigmoid(x) * (1 - sigmoid(x)).
inline float Sigmoid::_derivative(float value) const {
    return value * (1 - value);
}

// Sigmoid activation function squashes the output of each neuron to a value between 0 and 1.
std::vector<float> Sigmoid::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    auto output = _forward(input);

    // Apply the sigmoid function to the output of the layer.
    for (std::size_t i = 0; i < output.size(); i++) {
        output[i] = 1 / (1 + std::exp(-output[i]));
    }

    return output;
}

// ReLU activation function has a derivative of 1 if the output is greater than 0, otherwise 0.
inline float ReLU::_derivative(float value) const {
    return value > 0 ? 1 : 0;
}

// ReLU activation function squashes the output of each neuron to a value between 0 and infinity.
std::vector<float> ReLU::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    auto output = _forward(input);

    // Apply the ReLU function to the output of the layer.
    for (std::size_t i = 0; i < output.size(); i++) {
        output[i] = std::fmax(0, output[i]);
    }

    return output;
}

// Tanh activation function has a derivative of 1 - tanh(x)^2.
inline float Tanh::_derivative(float value) const {
    return 1 - std::pow(std::tanh(value), 2);
}

// Tanh activation function squashes the output of each neuron to a value between -1 and 1.
std::vector<float> Tanh::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    auto output = _forward(input);

    // Apply the tanh function to the output of the layer.
    for (std::size_t i = 0; i < output.size(); i++) {
        output[i] = std::tanh(output[i]);
    }

    return output;
}

// Softmax activation function has a derivative of softmax(x) * (1 - softmax(x)).
inline float Softmax::_derivative(float value) const {
    return value * (1 - value);
}

// Softmax activation function squashes the output of each neuron to a value between 0 and 1.
std::vector<float> Softmax::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    auto output = _forward(input);

    float sum = 0.0f;

    // Apply the softmax function to the output of the layer.
    for (std::size_t i = 0; i < output.size(); i++) {
        sum += std::exp(output[i]);
    }

    for (std::size_t i = 0; i < output.size(); i++) {
        output[i] = std::exp(output[i]) / sum;
    }

    return output;
}