#include <memory>
#include <stdexcept>
#include <vector>

#include "sgd.hpp"

inline void SGD::_step_layer(Layer& layer) {
    std::vector<std::vector<float>>& weights = layer.get_weights();
    std::vector<float>& biases = layer.get_biases();
    std::vector<std::vector<float>> weight_gradient = layer.get_weight_gradient();
    std::vector<float> bias_gradient = layer.get_bias_gradient();
    std::vector<std::vector<float>>& velocity = layer.get_velocity();

    // Update the weights and biases of the layer. The delta is the negative of the gradient
    // times the learning rate.
    for (std::size_t output_idx = 0; output_idx < layer.OUTPUT_SIZE; output_idx++) {
        for (std::size_t input_idx = 0; input_idx < layer.INPUT_SIZE; input_idx++) {
            // Decay the gradient to prevent overfitting by penalizing large weights.
            // I learned how to do this from https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab.
            float decayed_gradient = weight_gradient[input_idx][output_idx] + _weight_decay * weights[input_idx][output_idx];
            // Velocity is used to smooth the gradient by moving past local minima.
            // A velocity is stored for each weight. I learned how to do this from
            // https://towardsdatascience.com/gradient-descent-with-momentum-59420f626c8f#:~:text=To%20account%20for%20the%20momentum,moving%20average%20over%20these%20gradients.
            velocity[input_idx][output_idx] = _momentum * velocity[input_idx][output_idx] + _learning_rate * decayed_gradient;
            weights[input_idx][output_idx] -= velocity[input_idx][output_idx] + _weight_decay * weights[input_idx][output_idx];
        }

        biases[output_idx] -= _learning_rate * bias_gradient[output_idx];
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