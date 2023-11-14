#pragma once

#include <memory>
#include <vector>

#include "layer.hpp"

class FullyConnectedNN {
private:
    std::vector<std::unique_ptr<Layer>> _layers;

public:
    // Inserts a layer into the network. If there are already layers in the network,
    // the input size of the new layer must match the output size of the previous layer.
    void insert_layer(std::unique_ptr<Layer> layer);

    // Forward pass through the network. Returns the output of the network.
    std::vector<float> operator()(const std::vector<float>& input) const;
    // Forward pass through the network. Returns the index of the output neuron with the highest output value.
    int predict(const std::vector<float>& input) const;

    void backward(const std::vector<float>& expected_output) const;
    void backward(int expected_output_label) const;

    std::vector<std::unique_ptr<Layer>>& get_layers() {
        return _layers;
    }
};