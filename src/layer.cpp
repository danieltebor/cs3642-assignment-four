#include <math.h>
#include <random>
#include <stdexcept>

#include "layer.hpp"

Layer::Layer(std::size_t input_size, std::size_t output_size, std::function<float(float)> prev_activation_derivative)
    : INPUT_SIZE(input_size),
      OUTPUT_SIZE(output_size),
      _weights(input_size * output_size),
      _biases(output_size),
      _weight_gradient(input_size),
      _bias_gradient(output_size),
      _input(input_size),
      _output(output_size),
      _prev_activation_derivative(prev_activation_derivative) {
    // Reject a layer with 0 inputs or 0 outputs.
    if (input_size == 0 || output_size == 0) {
        throw std::invalid_argument("The number of neurons and connections per neuron must be greater than 0.");
    }

    // Initialize the weights to random values scaled by sqrt(2 / input_size).
    // This is called Xavier/Glorot Initialization.
    // https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / input_size));
    for (unsigned int i = 0; i < input_size * output_size; i++) {
        _weights[i] = (float)std::rand() / RAND_MAX * 2 - 1;
    }

    // Initialize the biases to random values between -1 and 1.
    for (unsigned int i = 0; i < output_size; i++) {
        _biases[i] = 0.0f;
    }
}

void Layer::_forward(const std::vector<float>& input) {
    // Calculate the dot product of each input and weight to the corresponding output.
    for (std::size_t output_idx = 0; output_idx < OUTPUT_SIZE; output_idx++) {
        float dot_product = 0.0f;

        for (std::size_t input_idx = 0; input_idx < INPUT_SIZE; input_idx++) {
            dot_product += input[input_idx] * _weights[input_idx * OUTPUT_SIZE + output_idx];

            // Store the input for the backward pass.
            if (output_idx == 0) {
                _input[input_idx] = input[input_idx];
            }
        }

        // Store the dot product plus the bias for the current output in the output vector.
        _output[output_idx] = dot_product + _biases[output_idx];
    }
}

std::vector<float> Layer::backward(const std::vector<float>& forward_gradient) {
    // Calculate the dot product of each input and weight to the corresponding output gradient.
    for (std::size_t input_idx = 0; input_idx < INPUT_SIZE; input_idx++) {
        float dot_product = 0.0f;

        for (std::size_t gradient_idx = 0; gradient_idx < OUTPUT_SIZE; gradient_idx++) {
            dot_product += forward_gradient[gradient_idx] * _weights[input_idx * OUTPUT_SIZE + gradient_idx];

            // Store the bias gradient for the optimizer.
            if (input_idx == 0) {
                _bias_gradient[gradient_idx] = forward_gradient[gradient_idx];
            }
        }

        // Compute partial derivative and store the weight gradient for the optimizer.
        _weight_gradient[input_idx] = dot_product * _prev_activation_derivative(_input[input_idx]);
    }

    // Store the gradients of the layer for the optimizer.
    return _weight_gradient;
}

std::vector<float> Layer::output_layer_backward(const std::vector<float>& expected_output) {
    // Calculate the gradient of the loss function with respect to the output of the layer.
    std::vector<float> output_gradient(OUTPUT_SIZE);
    for (std::size_t output_idx = 0; output_idx < OUTPUT_SIZE; output_idx++) {
        output_gradient[output_idx] = _loss_derivative(_output[output_idx], expected_output[output_idx]);
    }

    return backward(output_gradient);
}

// Assumes that the loss function is the mean squared error.
inline float Linear::_loss_derivative(float value, float expected_value) const {
    return (value - expected_value);
}

inline float Linear::activation_derivative(float value) const {
    return 1.0f;
}

std::vector<float> Linear::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    _forward(input);

    return _output;
}

// Assumes that the loss function is the mean squared error.
inline float Sigmoid::_loss_derivative(float value, float expected_value) const {
    return (value - expected_value) * activation_derivative(value);
}

inline float sigmoid(float value) {
    return 1 / (1 + std::exp(-value));
}

inline float Sigmoid::activation_derivative(float value) const {
    return sigmoid(value) * (1 - sigmoid(value));
}

std::vector<float> Sigmoid::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    _forward(input);

    // Apply the sigmoid function to the output of the layer.
    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
        _output[i] = sigmoid(_output[i]);
    }

    return _output;
}

// Assumes that the loss function is the mean squared error.
inline float ReLU::_loss_derivative(float value, float expected_value) const {
    return (value - expected_value) * activation_derivative(value);
}

inline float ReLU::activation_derivative(float value) const {
    return value > 0 ? 1 : 0;
}

std::vector<float> ReLU::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    _forward(input);

    // Apply the ReLU function to the output of the layer.
    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
        _output[i] = std::fmax(0, _output[i]);
    }

    return _output;
}

// Assumes that the loss function is the cross entropy loss.
inline float Softmax::_loss_derivative(float value, float expected_value) const {
    return expected_value - value;
}

inline float Softmax::activation_derivative(float value) const {
    throw std::logic_error("The softmax layer does not have an activation function and cannot be used for hidden layers.");
}

std::vector<float> Softmax::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    _forward(input);

    float sum = 0.0f;

    // Apply the softmax function to the output of the layer.
    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
        sum += std::exp(_output[i]);
    }

    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
        _output[i] = std::exp(_output[i]) / sum;
    }

    return _output;
}