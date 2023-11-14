#include <memory>
#include <stdexcept>
#include <vector>

#include "sgd.hpp"

inline void SGD::_step_layer(Layer& layer) {
    std::vector<float>& weights = layer.get_weights();
    std::vector<float>& biases = layer.get_biases();
    std::vector<float> weight_gradient = layer.get_weight_gradient();
    std::vector<float> bias_gradient = layer.get_bias_gradient();

    // Update the weights and biases of the layer. The delta is the negative of the gradient
    // times the learning rate.
    for (std::size_t input_idx = 0; input_idx < layer.INPUT_SIZE; input_idx++) {
        for (std::size_t output_idx = 0; output_idx < layer.OUTPUT_SIZE; output_idx++) {
            float delta = _learning_rate * weight_gradient[input_idx];
                //+ _momentum * weight_gradient[neuron_idx]
                //+ _weight_decay * weights[connection_idx * input_size + neuron_idx];

            weights[output_idx * layer.INPUT_SIZE + input_idx] -= delta;

            if (input_idx == 0) {
                biases[output_idx] -= _learning_rate * bias_gradient[output_idx];
            }
        }
    }
}

void SGD::step(FullyConnectedNN& network) {
    std::vector<std::unique_ptr<Layer>>& layers = network.get_layers();

    if (layers.empty()) {
        throw std::invalid_argument("The network must have at least one layer.");
    }

    for (auto& layer: layers) {
        _step_layer(*layer);
    }
}