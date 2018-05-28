#include <ctime>
#include <iostream>

#include <simple_nn/neural_net.hpp>

double mse_loss(const simple_nn::NeuralNet& network, const Eigen::MatrixXd& input, const Eigen::MatrixXd& output)
{
    // compute MSE
    Eigen::MatrixXd y = network.forward(input);
    double mse = 0.;
    for (int i = 0; i < y.cols(); i++) {
        mse += (y.col(i) - output.col(i)).squaredNorm();
    }
    mse /= double(y.cols());

    return mse;
}

int main()
{
    std::srand(std::time(NULL));
    // simple 1D regression
    // generate 50 random data in [-5,5]
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, 200).array() * 5.;
    // function is linear combination
    Eigen::MatrixXd output = input.array().cos();

    // Let's create our neural network
    simple_nn::NeuralNet network;
    // 1 hidden layer with 20 unites and sigmoid activation function
    network.add_layer<simple_nn::TanhLayer>(1, 20);
    // 1 output layer with sigmoid activation function
    network.add_layer<simple_nn::TanhLayer>(20, 1);

    // Random initial weights
    Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights());
    network.set_weights(theta);

    std::cout << "Initial MSE: " << mse_loss(network, input, output) << std::endl;

    // let's do an optimization
    // 1000 iterations/epochs
    int epochs = 1000;
    // learning rate
    double eta = 0.001;

    for (int i = 0; i < epochs; i++) {
        // get gradients
        Eigen::VectorXd dtheta = network.backward(input, output);

        // update weights
        theta = theta.array() - eta * dtheta.array();
        network.set_weights(theta);

        if (i % 100 == 0) {
            std::cout << "MSE: " << mse_loss(network, input, output) << std::endl;
        }
    }

    std::cout << "Final MSE: " << mse_loss(network, input, output) << std::endl;

    return 0;
}