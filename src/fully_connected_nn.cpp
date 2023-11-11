#include <iostream>

#include "fully_connected_nn.hpp"

// Insert a layer into the network.
// This allows a custom topology to be created.
void FullyConnectedNN::insert_layer(std::unique_ptr<Layer> layer) {
    if (_layers.size() != 0) {
        // Throw an error if the input size of the new layer does not match the output size of the previous layer.
        std::size_t prev_layer_output_size = _layers[_layers.size() - 1]->get_num_connections_per_neuron();
        std::size_t new_layer_input_size = layer->get_num_neurons();

        if (prev_layer_output_size != new_layer_input_size) {
            throw std::invalid_argument("The input size of the new layer must match the output size of the previous layer.");
        }
    }

    _layers.push_back(std::move(layer));
}

// Forward pass through the network.
// The inputs vector is the input to the network and must be the same size as the number
// of neurons in the first layer. Returns the outputs of each layer in the network.
std::vector<float> FullyConnectedNN::operator()(const std::vector<float>& input) const {
    auto current_input = input;
    
    // Forward pass through each layer.
    for (auto& layer : _layers) {
        // Forward pass through the current layer.
        auto current_output = (*layer)(current_input);

        // Set the input of the next layer to the output of the current layer.
        current_input = current_output;
    }

    return current_input;
}

// Predict the output of the network.
// The inputs vector is the input to the network and must be the same size as the number
// of input neurons in the first layer. Returns the index of the output neuron with the highest
// output value.
int FullyConnectedNN::predict(const std::vector<float>& input) const {
    // Forward pass through the network.
    auto output = (*this)(input);

    // Find the index of the output neuron with the highest output value.
    int max_output_idx = 0;
    float max_output_value = output[0];
    for (std::size_t output_idx = 1; output_idx < output.size(); output_idx++) {
        if (output[output_idx] > max_output_value) {
            max_output_idx = output_idx;
            max_output_value = output[output_idx];
        }
    }

    return max_output_idx;
}

// Backward pass through the network.
// The output_loss_gradient vector is the gradient calculated by the loss function.
// The gradient is calculated for each layer in the network using the gradient of the
// previous layer. This applies the chain rule of calculus.
void FullyConnectedNN::backward(const std::vector<float>& output_loss_gradient) const {
    auto current_gradient = output_loss_gradient;

    // Backward pass through each layer.
    for (std::size_t i = _layers.size(); i > 0; i--) {
        // Backward pass through the current layer.
        current_gradient = _layers[i - 1]->backward(current_gradient);
    }
}