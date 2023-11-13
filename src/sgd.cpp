#include <memory>
#include <stdexcept>
#include <vector>

#include "sgd.hpp"

// Update the weights and biases of a layer.
inline void SGD::_step_layer(Layer& layer) {
    std::vector<float>& weights = layer.get_weights();
    std::vector<float>& biases = layer.get_biases();
    std::vector<float>& weight_gradient = layer.get_current_weight_gradient();
    std::vector<float>& bias_gradient = layer.get_current_bias_gradient();
    std::size_t num_neurons = layer.get_num_neurons();
    std::size_t num_connections_per_neuron = layer.get_num_connections_per_neuron();

    if (weight_gradient.empty()) {
        throw std::invalid_argument("The layer have a backward pass called before the optimizer step.");
    }

    // Update the weights and biases of the layer.
    // The weights and biases are updated using the gradients of the layer.
    // The gradients are calculated during the backward pass.
    // The learning rate is the step size of the optimizer.
    // A higher learning rate will result in faster training, but the model may not converge.
    // The momentum is used to smooth the gradient by moving past local minima.
    // A higher momentum will result in faster training, but the model may not converge.
    // Weight decay is used to prevent overfitting by penalizing large weights.
    // A higher weight decay will result in a simpler model, but the model may not converge.
    for (std::size_t i = 0; i < num_connections_per_neuron; i++) {
        for (std::size_t j = 0; j < num_neurons; j++) {
            float delta = _learning_rate * weight_gradient[j];
                + _momentum * weight_gradient[j]
                + _weight_decay * weights[j * num_connections_per_neuron + i];

            weights[j * num_connections_per_neuron + i] -= delta;
        }
        biases[i] -= bias_gradient[i];
    }
}

// Update the weights and biases of the network.
void SGD::step(FullyConnectedNN& network) {
    std::vector<std::unique_ptr<Layer>>& layers = network.get_layers();

    if (layers.empty()) {
        throw std::invalid_argument("The network must have at least one layer.");
    }

    for (auto& layer: layers) {
        _step_layer(*layer);
    }
}