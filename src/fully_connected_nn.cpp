#include <stdexcept>

#include "fully_connected_nn.hpp"

void FullyConnectedNN::insert_layer(std::unique_ptr<Layer> layer) {
    // Throw an error if the input size of the new layer does not match the output size of the previous layer.
    if (_layers.size() != 0) {
        if (_layers.back()->OUTPUT_SIZE != layer->INPUT_SIZE) {
            throw std::invalid_argument("The input size of the new layer must match the output size of the previous layer.");
        }

        // Set the previous activation derivative of the new layer to the activation derivative of the previous layer.
        std::function<float(float)> activation_derivative_func = std::bind(&Layer::activation_derivative, _layers.back().get(), std::placeholders::_1);
        layer->set_prev_activation_derivative(activation_derivative_func);
    }

    _layers.push_back(std::move(layer));
}

std::vector<float> FullyConnectedNN::operator()(const std::vector<float>& input) const {
    if (_layers.size() == 0) {
        throw std::invalid_argument("The network must have at least one layer.");
    }
    else if (input.size() != _layers[0]->INPUT_SIZE) {
        throw std::invalid_argument("The input size must match the number of neurons in the first layer.");
    }

    auto current_input = input;
    
    // Forward pass through each layer.
    for (auto& layer : _layers) {
        // Forward pass through the current layer.
        current_input = (*layer)(current_input);
    }

    return current_input;
}

int FullyConnectedNN::predict(const std::vector<float>& input) const {
    // Throw an error if the network has no layers or if the input size does not match the number of neurons in the first layer.
    if (_layers.size() == 0) {
        throw std::invalid_argument("The network must have at least one layer.");
    }
    else if (input.size() != _layers[0]->INPUT_SIZE) {
        throw std::invalid_argument("The input size must match the number of neurons in the first layer.");
    }

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

void FullyConnectedNN::backward(const std::vector<float>& expected_output) const {
    // Throw an error if the network has no layers or if the expected output size does not match the number of neurons in the last layer.
    if (_layers.size() == 0) {
        throw std::invalid_argument("The network must have at least one layer.");
    }
    else if (expected_output.size() != _layers[_layers.size() - 1]->OUTPUT_SIZE) {
        throw std::invalid_argument("The expected output size must match the number of neurons in the last layer.");
    }

    std::vector<float> current_gradient = _layers[_layers.size() - 1]->output_layer_backward(expected_output);

    if (_layers.size() == 1) {
        return;
    }

    // Backward pass through each layer.
    for (std::size_t i = _layers.size() - 1; i > 0; i--) {
        // Backward pass through the current layer.
        current_gradient = _layers[i - 1]->backward(current_gradient);
    }
}

void FullyConnectedNN::backward(int expected_output_label) const {
    // Throw an error if the network has no layers or if the expected output label is out of range.
    if (_layers.size() == 0) {
        throw std::invalid_argument("The network must have at least one layer.");
    }

    std::size_t output_size = _layers[_layers.size() - 1]->OUTPUT_SIZE;

    if (expected_output_label < 0 || expected_output_label >= output_size) {
        throw std::invalid_argument("Expected output label is out of range.");
    }

    // Convert the expected output label to a vector of floats.
    std::vector<float> expected_output(output_size, 0.0f);
    expected_output[expected_output_label] = 1.0f;

    backward(expected_output);
}