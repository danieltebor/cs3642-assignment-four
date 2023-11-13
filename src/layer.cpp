#include <math.h>
#include <stdexcept>

#include "layer.hpp"

// The constructor takes the number of number of neurons, the number of connections that each
// neuron has to the next layer, and the activation function for the layer outputs.
// The weights vector is initialized to the size of the number of neurons in the current layer
// times the number of neurons in the next layer. The biases vector is initialized to the number
// of connections per neuron to the next layer. The weights are initialized to random
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
        _biases[i] = 0.0f;
    }
}

// Forward pass through the layer.
// The input vector is the input to the layer and must be the same size as the 
// layer input size. Returns the output of the layer.
std::vector<float> Layer::_forward(const std::vector<float>& input) {
    // The output of the layer.
    std::vector<float> output(_num_connections_per_neuron);
    
    // Calculate each output value of the layer.
    for (std::size_t connection_idx = 0; connection_idx < _num_connections_per_neuron; connection_idx++) {
        // The output of the current neuron.
        float current_output_value = 0.0f;

        for (std::size_t neuron_idx = 0; neuron_idx < _num_neurons; neuron_idx++) {
            // Add current input value * the weight for the current connection.
            current_output_value += input[connection_idx] * _weights[connection_idx * _num_neurons + neuron_idx];
        }

        // Add the bias for the current neuron.
        current_output_value += _biases[connection_idx];

        // Add the output of the current neuron to the output of the current layer.
        output[connection_idx] = current_output_value;
    }

    // Store the output of the layer for the backward pass.
    _current_output = output;
    return output;
}

// Backward pass through the network.
// The prev_gradient vector is the gradient calculated by the loss function.
// The gradient is calculated for each layer in the network using the gradient of the
// previous layer. This applies the chain rule of calculus.
std::vector<float> Layer::backward(const std::vector<float>& prev_gradient) {
    std::vector<float> output_gradient(_num_neurons);
    std::vector<float> bias_gradient(_num_connections_per_neuron);

    // Calculate the gradient of each neuron in the layer.
    // The gradient of each neuron is the sum of the gradient of each connection to the next layer
    // multiplied by the weight of the connection
    for (std::size_t neuron_idx = 0; neuron_idx < _num_neurons; neuron_idx++) {
        float sum = 0.0f;
        for (std::size_t connection_idx = 0; connection_idx < _num_connections_per_neuron; connection_idx++) {
            sum += prev_gradient[connection_idx] * _weights[neuron_idx * _num_connections_per_neuron + connection_idx];

            // Ensures values in bias_gradient are updated atomically.
            bias_gradient[connection_idx] += prev_gradient[connection_idx];
        }
        output_gradient[neuron_idx] = sum * _derivative(_current_output[neuron_idx]);
    }

    // Store the gradients of the layer for the optimizer.
    _current_weight_gradient = output_gradient;
    _current_bias_gradient = bias_gradient;
    return output_gradient;
}

// The output_layer_backward is called on the last layer of the network. It applies
// a derivative with respect to the loss function.
std::vector<float> Layer::output_layer_backward(const std::vector<float>& expected_output) {

}

// Calling the class as a function executes a forward pass through the layer.

// Linear activation function has a derivative of 1 since it does not use an activation function on the output.
inline float Linear::_derivative(float value) const {
    return 1.0f;
}

//
// Linear assumes that mean squared error is used as the loss function.
inline float Linear::_loss_derivative(float value) const {

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

//
inline float Sigmoid::_loss_derivative(float value) const {

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

// 
inline float ReLU::_loss_derivative(float value) const {

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

// The softmax derivative is set to 1. This is not actually the derivative of softmax,
// but in this case the gradient from the activation function is all that is needed.
inline float Softmax::_derivative(float value) const {
    return 1.0f;
}

//
// Softmax assumes that cross entropy is the loss function.
inline float Softmax::_loss_derivative(float value) const {

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