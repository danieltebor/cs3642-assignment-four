#include <math.h>
#include <stdexcept>

#include "loss_function.hpp"

float LossFunction::operator()(const std::vector<float>& output, int expected_output_label) const {
    // Throw an error if the expected output label is out of range.
    if (expected_output_label < 0 || expected_output_label >= output.size()) {
        throw std::invalid_argument("Expected output label is out of range.");
    }

    // Convert the expected output label to a vector of floats.
    std::vector<float> expected_output(output.size(), 0.0f);
    expected_output[expected_output_label] = 1.0f;

    return (*this)(output, expected_output);
}

float MeanSquaredError::operator()(const std::vector<float>& output, const std::vector<float>& expected_output) const {
    // Throw an error if the output and expected output are not the same size.
    if (output.size() != expected_output.size()) {
        throw std::invalid_argument("The output and expected output must be the same size.");
    }

    float sum = 0.0f;

    // Calculate the mean squared error for each output.
    for (std::size_t i = 0; i < output.size(); i++) {
        sum += std::pow(output[i] - expected_output[i], 2);
    }

    // Return the mean of the mean squared errors.
    return sum / output.size();
}

float CrossEntropyLoss::operator()(const std::vector<float>& output, const std::vector<float>& expected_output) const {
    // Throw an error if the output and expected output are not the same size.
    if (output.size() != expected_output.size()) {
        throw std::invalid_argument("The output and expected output must be the same size.");
    }

    float sum = 0.0f;

    // Calculate the cross entropy for each output.
    for (std::size_t i = 0; i < output.size(); i++) {
        sum += expected_output[i] * std::log(output[i]);
    }

    // Return the negative of the sum of the cross entropies.
    return -sum;
}