#include <math.h>
#include <stdexcept>

#include "loss_function.hpp"

// Converts the expected output label to a vector of floats of all 0s except for the index of the
// expected output label, which is 1. The expected output label must be in the range of the output
// vector. This vector is then passed to the other operator() function.
float LossFunction::operator()(const std::vector<float>& output, int expected_output_label) const {
    if (expected_output_label < 0 || expected_output_label >= output.size()) {
        throw std::invalid_argument("Expected output label is out of range.");
    }

    // Convert the expected output label to a vector of floats.
    std::vector<float> expected_output(output.size(), 0.0f);
    expected_output[expected_output_label] = 1.0f;

    return (*this)(output, expected_output);
}

// Converts the expected output label to a vector of floats of all 0s except for the index of the
// expected output label, which is 1. The expected output label must be in the range of the output
// vector. This vector is then passed to the other derivative() function.
#include <iostream>
std::vector<float> LossFunction::derivative(const std::vector<float>& output, int expected_output_label) const {
    if (expected_output_label < 0 || expected_output_label >= output.size()) {
        throw std::invalid_argument("Expected output label is out of range.");
    }

    // Convert the expected output label to a vector of floats.
    std::vector<float> expected_output(output.size(), 0.0f);
    expected_output[expected_output_label] = 1.0f;

    return derivative(output, expected_output);
}

// Mean squared error returns the mean of the summed squared errors.
float MeanSquaredError::operator()(const std::vector<float>& output, const std::vector<float>& expected_output) const {
    if (output.size() != expected_output.size()) {
        throw std::invalid_argument("The output and expected output must be the same size.");
    }

    float loss = 0.0f;

    // Calculate the mean squared error for each output.
    for (std::size_t i = 0; i < output.size(); i++) {
        loss += std::pow(output[i] - expected_output[i], 2);
    }

    // Return the mean of the mean squared errors.
    return loss / output.size();
}

// MeanSquaredError loss function has a derivative of 2 * (output - expected_output). The derivative
// is averaged over the number of outputs so that the magnitude of the gradient is not dependent on 
// the number of outputs.
std::vector<float> MeanSquaredError::derivative(const std::vector<float>& output, const std::vector<float>& expected_output) const {
    if (output.size() != expected_output.size()) {
        throw std::invalid_argument("The output and expected output must be the same size.");
    }

    std::vector<float> loss_gradient(output.size());

    // Calculate the derivative of the mean squared error function for each output.
    for (std::size_t i = 0; i < output.size(); i++) {
        loss_gradient[i] = 2.0f / output.size() * (output[i] - expected_output[i]);
    }

    return loss_gradient;
}

// Cross entropy returns the negative of the sum of the expected output multiplied by the natural
// log of the output.
float CrossEntropyLoss::operator()(const std::vector<float>& output, const std::vector<float>& expected_output) const {
    if (output.size() != expected_output.size()) {
        throw std::invalid_argument("The output and expected output must be the same size.");
    }

    float loss = 0.0f;

    // Calculate the cross entropy for each output.
    for (std::size_t i = 0; i < output.size(); i++) {
        loss += expected_output[i] * std::log(output[i]);
    }

    // Return the negative of the sum of the cross entropies.
    return -loss;
}

// CrossEntropy loss function has a derivative of output - expected_output. This function assumes
// that the output has already been passed through the softmax activation function.
std::vector<float> CrossEntropyLoss::derivative(const std::vector<float>& output, const std::vector<float>& expected_output) const {
    if (output.size() != expected_output.size()) {
        throw std::invalid_argument("The output and expected output must be the same size.");
    }

    std::vector<float> loss_gradient(output.size());

    for (std::size_t i = 0; i < output.size(); i++) {
        loss_gradient[i] = output[i] - expected_output[i];
    }

    return loss_gradient;
}