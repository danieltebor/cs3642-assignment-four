#pragma once

#include <memory>
#include <vector>

#include "layer.hpp"

#include <iostream>
class FullyConnectedNN {
private:
    std::vector<std::unique_ptr<Layer>> _layers;

public:
    // Insert a layer into the network.
    // This allows a custom topology to be created.
    void insert_layer(std::unique_ptr<Layer>);

    // Forward pass through the network.
    // The input vector is the input to the network and must be the same size as the number
    // of neurons in the first layer. Returns the output in the network.
    std::vector<float> operator()(const std::vector<float>& input) const;

    // Predict the output of the network.
    // The input vector is the input to the network and must be the same size as the number
    // of neurons in the first layer. Returns the index of the output neuron with the highest
    // output value.
    int predict(const std::vector<float>& input) const;

    // Backward pass through the network.
    // The output_loss_gradient vector is the gradient calculated by the loss function.
    // The gradient is calculated for each layer in the network using the gradient of the
    // previous layer. This applies the chain rule of calculus.
    // Returns the gradient of the network.
    void backward(const std::vector<float>& output_loss_gradient) const;

    // Get the layers of the network.
    std::vector<std::unique_ptr<Layer>>& get_layers() {
        return _layers;
    }
};