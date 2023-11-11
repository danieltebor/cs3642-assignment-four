#pragma once

#include "fully_connected_nn.hpp"

// The stochastic gradient descent optimizer is used to update the weights and biases of the network.
// It works by calculating utilizing the gradients of the network to update the weights and biases.
class SGD {
private:
    // The learning rate is the step size of the optimizer.
    // A higher learning rate will result in faster training, but the model may not converge.
    float _learning_rate;
    // The momentum is used to smooth the gradient by moving past local minima.
    // A higher momentum will result in faster training, but the model may not converge.
    float _momentum;
    // Weight decay is used to prevent overfitting by penalizing large weights.
    // A higher weight decay will result in a simpler model, but the model may not converge.
    float _weight_decay;

    // Update the weights and biases of a layer.
    inline void _step_layer(Layer& layer);

public:
    // The constructor takes the learning rate, momentum, and weight decay.
    // The default values are 0.0f for momentum and weight decay.
    SGD(float learning_rate, float momentum = 0.0f, float weight_decay = 0.0f)
        : _learning_rate(learning_rate), _momentum(momentum), _weight_decay(weight_decay) {};

    // Update the weights and biases of the network.
    void step(FullyConnectedNN& network);

    float get_learning_rate() const {
        return _learning_rate;
    }

    void set_learning_rate(float learning_rate) {
        _learning_rate = learning_rate;
    }

    float get_momentum() const {
        return _momentum;
    }

    void set_momentum(float momentum) {
        _momentum = momentum;
    }

    float get_weight_decay() const {
        return _weight_decay;
    }

    void set_weight_decay(float weight_decay) {
        _weight_decay = weight_decay;
    }
};