#pragma once

#include <vector>

// The loss function is used to calculate the error of the network.
class LossFunction {
public:
    // The loss function is used to calculate the error of the network.
    // The loss function takes the output of the network and the expected output, which
    // is either a vector of floats that represent the correct model output, or a single 
    // integer label that is the index of the correct model output.
    virtual float operator()(const std::vector<float>& output, const std::vector<float>& expected_output) const = 0;
    float operator()(const std::vector<float>& output, int expected_output_label) const;
};

// The mean squared error loss function is used for regression problems.
class MSELoss : public LossFunction {
public:
    using LossFunction::operator();

    float operator()(const std::vector<float>& output, const std::vector<float>& expected_output) const override;
};

// The cross entropy loss function is used for classification problems.
class CrossEntropyLoss : public LossFunction {
public:
    using LossFunction::operator();

    float operator()(const std::vector<float>& output, const std::vector<float>& expected_output) const override;
};