#pragma once

#include <vector>

struct Layer {
private:
    // The weights represent the connections between the current layer and the next layer.
    // The vector is flattened to a 1D array for better memory locality.
    std::vector<float> _weights;
    // The biases represent the bias of each neuron in the layer. The biases are
    // important because they allow for the output of a neuron to be non-zero even if
    // all of the inputs are zero.
    std::vector<float> _biases;

    // The input size of the layer.
    std::size_t _num_neurons;
    // The output size of the layer.
    std::size_t _num_connections_per_neuron;

protected:
    // The output of the layer for the backward pass.
    std::vector<float> _current_output = {};
    // The gradient of the layer for the optimizer.
    std::vector<float> _current_weight_gradient = {};
    // The gradient of the biases of the layer.
    std::vector<float> _current_bias_gradient = {};

    // Forward pass through the layer.
    // The input vector is the input to the layer and must be the same size as the 
    // layer input size. Returns the output of the layer.
    std::vector<float> _forward(const std::vector<float>& input);

    // The derivative of the activation function is used to calculate the derivative of each
    // value in the output.
    inline virtual float _derivative(float value) const = 0;

public:
    // The constructor takes the number of number of neurons, the number of connections that each
    // neuron has to the next layer, and the activation function for the layer outputs.
    // The weights vector is initialized to the size of the number of neurons in the current layer
    // times the number of neurons in the next layer. The biases vector is initialized to the number
    // of connections per neuron to the next layer. The weights and biases are initialized to random
    // values between -1 and 1.
    Layer(unsigned int num_neurons, unsigned int num_connections_per_neuron);

    virtual ~Layer() = default;

    // Forward pass through the layer.
    // Specific activation functions are implemented in the derived classes.
    virtual std::vector<float> operator()(const std::vector<float>& input) = 0;
    // Backward pass through the network.
    // The output_loss_gradient vector is the gradient calculated by the loss function.
    // The gradient is calculated for each layer in the network using the gradient of the
    // previous layer. This applies the chain rule of calculus.
    std::vector<float> backward(const std::vector<float>& prev_gradient);

    // Get the weights of the layer.
    std::vector<float>& get_weights() {
        return _weights;
    }

    // Get the biases of the layer.
    std::vector<float>& get_biases() {
        return _biases;
    }

    std::vector<float>& get_current_weight_gradient() {
        return _current_weight_gradient;
    }

    std::vector<float>& get_current_bias_gradient() {
        return _current_bias_gradient;
    }

    // Get the input size of the layer.
    std::size_t get_num_neurons() const {
        return _num_neurons;
    }

    // Get the output size of the layer.
    std::size_t get_num_connections_per_neuron() const {
        return _num_connections_per_neuron;
    }
};

// The derivative of the activation function is used to calculate the gradient of the layer
// during the backward pass.

// Calling the class as a function executes a forward pass through the layer.

// Calling backward() executes a backward pass through the layer.
// The prev_gradient vector is the loss gradient calculated from the previous layer.
// Returns the gradient of the layer.

// Linear activation function does not use an activation function on the output.
class Linear : public Layer {
protected:
    inline float _derivative(float value) const override;

public:
    Linear(unsigned int num_neurons, unsigned int num_connections_per_neuron) : Layer(num_neurons, num_connections_per_neuron) {}

    std::vector<float> operator()(const std::vector<float>& input) override;
};

// Sigmoid activation function squashes the output of each neuron to a value between 0 and 1.
class Sigmoid : public Layer {
protected:
    inline float _derivative(float value) const override;

public:
    Sigmoid(unsigned int num_neurons, unsigned int num_connections_per_neuron) : Layer(num_neurons, num_connections_per_neuron) {}

    std::vector<float> operator()(const std::vector<float>& input) override;
};

// ReLU activation function squashes the output of each neuron to a value between 0 and infinity.
class ReLU : public Layer {
protected:
    inline float _derivative(float value) const override;

public:
    ReLU(unsigned int num_neurons, unsigned int num_connections_per_neuron) : Layer(num_neurons, num_connections_per_neuron) {}

    std::vector<float> operator()(const std::vector<float>& input) override;
};

// Tanh activation function squashes the output of each neuron to a value between -1 and 1.
class Tanh : public Layer {
protected:
    inline float _derivative(float value) const override;

public:
    Tanh(unsigned int num_neurons, unsigned int num_connections_per_neuron) : Layer(num_neurons, num_connections_per_neuron) {}

    std::vector<float> operator()(const std::vector<float>& input) override;
};

// Softmax activation function squashes the output of each neuron to a value between 0 and 1.
class Softmax : public Layer {
protected:
    inline float _derivative(float value) const override;

public:
    Softmax(unsigned int num_neurons, unsigned int num_connections_per_neuron) : Layer(num_neurons, num_connections_per_neuron) {}

    std::vector<float> operator()(const std::vector<float>& input) override;
};