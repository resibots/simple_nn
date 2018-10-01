//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#include <ctime>
#include <iostream>

#include <simple_nn/loss.hpp>
#include <simple_nn/neural_net.hpp>

int main()
{
    std::srand(std::time(NULL));
    // simple 1D regression
    // generate 200 random data in [-5,5]
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, 200).array() * 5.;
    // function is linear combination
    Eigen::MatrixXd output = input.array().cos();

    // Let's create our neural network
    simple_nn::NeuralNet network;
    // 1 hidden layer with 20 unites and tanh activation function
    network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(1, 20);
    // 1 output layer with tanh activation function
    network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(20, 1);

    // Random initial weights
    Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights());
    network.set_weights(theta);

    std::cout << "Initial MSE: " << network.get_loss<simple_nn::MeanSquaredError>(input, output) << std::endl;

    // let's do an optimization
    // 1000 iterations/epochs
    int epochs = 1000;
    // learning rate
    double eta = 0.01;

    for (int i = 0; i < epochs; i++) {
        // get gradients
        Eigen::VectorXd dtheta = network.backward<simple_nn::MeanSquaredError>(input, output);

        // update weights
        theta = theta.array() - eta * dtheta.array();
        network.set_weights(theta);

        if (i % 100 == 0) {
            std::cout << "MSE: " << network.get_loss<simple_nn::MeanSquaredError>(input, output) << std::endl;
        }
    }

    std::cout << "Final MSE: " << network.get_loss<simple_nn::MeanSquaredError>(input, output) << std::endl;

    return 0;
}