#include <ctime>
#include <iostream>

#include <simple_nn/loss.hpp>
#include <simple_nn/neural_net.hpp>

int main()
{
    std::srand(std::time(NULL));
    // simple 1D regression
    // generate 200 random data in [-5,5]
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, 500).array() * 5.;
    // function is linear combination
    Eigen::MatrixXd output = input.array().cos();

    // Let's create our neural network
    simple_nn::NeuralNet network;
    // 1 hidden layer with 20 unites and sigmoid activation function
    network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Gaussian>>(1, 20);
    // 1 output layer with no activation function
    network.add_layer<simple_nn::FullyConnectedLayer<>>(20, 2);

    // Random initial weights
    Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights());
    network.set_weights(theta);

    std::cout << "Initial: " << network.get_loss<simple_nn::NegativeLogGaussianPrediction>(input, output) << std::endl;

    // let's do an optimization
    // 10000 iterations/epochs
    int epochs = 10000;
    // learning rate
    double eta = 0.00001;

    for (int i = 0; i < epochs; i++) {
        // get gradients
        Eigen::VectorXd dtheta = network.backward<simple_nn::NegativeLogGaussianPrediction>(input, output);

        // update weights
        theta = theta.array() - eta * dtheta.array();
        network.set_weights(theta);

        if (i % 1000 == 0) {
            std::cout << i << ": " << network.get_loss<simple_nn::NegativeLogGaussianPrediction>(input, output) << std::endl;
        }
    }

    std::cout << "Final: " << network.get_loss<simple_nn::NegativeLogGaussianPrediction>(input, output) << std::endl;

    Eigen::VectorXd query(1);
    query << 0.;

    Eigen::MatrixXd out = network.forward(query);

    std::cout << out(0, 0) << " " << std::exp(out(1, 0)) << std::endl;

    return 0;
}