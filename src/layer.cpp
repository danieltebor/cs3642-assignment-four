#include <math.h>
#include <random>
#include <stdexcept>

#include "layer.hpp"

Layer::Layer(std::size_t input_size, std::size_t output_size)
    : INPUT_SIZE(input_size),
      OUTPUT_SIZE(output_size),
      _weights(input_size, std::vector<float>(output_size)),
      _biases(output_size),
      _input(input_size),
      _output(output_size),
      _weight_gradient(input_size, std::vector<float>(output_size)),
      _error_gradient(input_size),
      _bias_gradient(output_size),
      _velocity(input_size, std::vector<float>(output_size)) {
    // Reject a layer with 0 inputs or 0 outputs.
    if (input_size == 0 || output_size == 0) {
        throw std::invalid_argument("The number of neurons and connections per neuron must be greater than 0.");
    }

    // Initialize the weights to random values scaled by sqrt(2 / input_size).
    // This is called Xavier/Glorot Initialization.
    // https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / input_size));
    for (std::size_t input_idx = 0; input_idx < input_size; input_idx++) {
        for (std::size_t output_idx = 0; output_idx < output_size; output_idx++) {
            _weights[input_idx][output_idx] = distribution(generator);
        }
    }

    // Initialize the biases to random values between -1 and 1.
    for (unsigned int i = 0; i < output_size; i++) {
        _biases[i] = 0.0f;
    }
}

void Layer::_forward(const std::vector<float>& input) {
    // Calculate the weighted sum of the inputs and store the result in an output vector.
    for (std::size_t output_idx = 0; output_idx < OUTPUT_SIZE; output_idx++) {
        float weighted_sum = 0.0f;

        for (std::size_t input_idx = 0; input_idx < INPUT_SIZE; input_idx++) {
            weighted_sum += input[input_idx] * _weights[input_idx][output_idx];
        }

        // Store the dot product plus the bias for the current output in the output vector
        _output[output_idx] = weighted_sum + _biases[output_idx];
    }

    // Store the input for the backward pass.
    for (std::size_t input_idx = 0; input_idx < INPUT_SIZE; input_idx++) {
        _input[input_idx] = input[input_idx];
    }
}

std::vector<float> Layer::_backward(const std::vector<float>& error_gradient) {
    // Compute the gradient for each weight and bias.
    for (std::size_t input_idx = 0; input_idx < INPUT_SIZE; input_idx++) {
        float weighted_sum = 0.0f;

        for (std::size_t derivative_idx = 0; derivative_idx < OUTPUT_SIZE; derivative_idx++) {
            // Compute the dot product of the error gradient and the weights.
            _weight_gradient[input_idx][derivative_idx] = error_gradient[derivative_idx] * _input[input_idx];
            // Compute the weighted sum of the error gradient.
            weighted_sum += error_gradient[derivative_idx] * _weights[input_idx][derivative_idx];
        }

        // Compute partial derivative and store the weight gradient for the optimizer.
        _error_gradient[input_idx] = weighted_sum;
    }

    // Set the bias gradient to the error gradient.
    for (std::size_t derivative_idx = 0; derivative_idx < OUTPUT_SIZE; derivative_idx++) {
        _bias_gradient[derivative_idx] = error_gradient[derivative_idx];
    }

    return _error_gradient;
}

std::vector<float> Layer::backward(std::vector<float>& error_gradient) {
    // Calculate the gradient of the activation function with respect to error of the previous layer/s.
    for (std::size_t output_idx = 0; output_idx < OUTPUT_SIZE; output_idx++) {
        error_gradient[output_idx] = _activation_derivative(_output[output_idx]) * error_gradient[output_idx];
    }

    return _backward(error_gradient);
}

std::vector<float> Layer::output_layer_backward(const std::vector<float>& expected_output) {
    // Calculate the gradient of the loss function with respect to the output of the layer.
    std::vector<float> error_gradient(OUTPUT_SIZE);
    for (std::size_t output_idx = 0; output_idx < OUTPUT_SIZE; output_idx++) {
        error_gradient[output_idx] = _loss_derivative(_output[output_idx], expected_output[output_idx]);
    }

    return _backward(error_gradient);
}

inline float Linear::_activation_derivative(float value) const {
    return 1.0f;
}

// Assumes that the loss function is the mean squared error.
inline float Linear::_loss_derivative(float predicted_value, float expected_value) const {
    // The 2 from the derivative of MSE is ommitted because it is a scalar and does not affect the direction of the gradient.
    return (predicted_value - expected_value) / OUTPUT_SIZE;
}

std::vector<float> Linear::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    _forward(input);

    return _output;
}

inline float Sigmoid::_activation_derivative(float value) const {
    return value * (1 - value);
}

// Assumes that the loss function is the mean squared error.
inline float Sigmoid::_loss_derivative(float predicted_value, float expected_value) const {
    // The 2 from the derivative of MSE is ommitted because it is a scalar and does not affect the direction of the gradient.
    return predicted_value * (1 - predicted_value) * (predicted_value - expected_value) / OUTPUT_SIZE;
}

std::vector<float> Sigmoid::operator()(const std::vector<float>& input) {
    // Pass the input through the layer.
    _forward(input);

    // Apply the sigmoid function to the output of the layer.
    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
        _output[i] = 1 / (1 + std::exp(-_output[i]));
    }

    return _output;
}

inline float ReLU::_activation_derivative(float value) const {
    return value > 0 ? 1 : 0;
}

// Assumes that the loss function is the mean squared error.
inline float ReLU::_loss_derivative(float predicted_value, float expected_value) const {
    // The 2 from the derivative of MSE is ommitted because it is a scalar and does not affect the direction of the gradient.
    return predicted_value * (predicted_value - expected_value) / OUTPUT_SIZE;
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

inline float Softmax::_activation_derivative(float value) const {
    throw std::logic_error("The softmax layer does not have an implemented activation derivative and cannot be used for hidden layers.");
}

// Assumes that the loss function is the cross entropy loss.
inline float Softmax::_loss_derivative(float predicted_value, float expected_value) const {
    return predicted_value - expected_value;
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