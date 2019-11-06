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
#include <fstream>
#include <iostream>

#include <simple_nn/loss.hpp>
#include <simple_nn/neural_net.hpp>
#include <simple_nn/opt.hpp>

int main()
{
    std::srand(std::time(NULL));
    // simple 1D regression
    // generate 200 random data in [-5,5]
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, 200).array() * 5.;
    // function is linear combination
    Eigen::MatrixXd output = input.array().cos().array() + Eigen::MatrixXd::Random(1, 200).array() * 0.2;

    // Let's create our neural network
    simple_nn::NeuralNet network;
    // 1 hidden layer with 20 units and tanh activation function
    network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(1, 20);
    // 1 output layer with no activation function
    network.add_layer<simple_nn::FullyConnectedLayer<>>(20, 2);

    // Random initial weights
    Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights()).array() * std::sqrt(1 / 20.);
    network.set_weights(theta);

    std::cout << "Initial: " << network.get_loss<simple_nn::NegativeLogGaussianPrediction<>>(input, output) << std::endl;

    // let's do an optimization
    // 10000 iterations/epochs
    int epochs = 10000;
    // Adam optimizer
    simple_nn::Adam optimizer;
    optimizer.reset(theta);

    // This is a functor that returns (value, gradient)
    auto eval = [&](const Eigen::VectorXd& params) {
        network.set_weights(params);

        // get gradients
        Eigen::VectorXd dtheta = network.backward<simple_nn::NegativeLogGaussianPrediction<>>(input, output);

        // Adam does not care about the value, so we do not spend time in computing it and return 0.
        return std::make_pair(0., dtheta);
    };

    for (int i = 0; i < epochs; i++) {

        bool stop;
        std::tie(stop, std::ignore, theta) = optimizer.optimize_once(eval);
        network.set_weights(theta);

        if (i % 100 == 0) {
            std::cout << i << ": " << network.get_loss<simple_nn::NegativeLogGaussianPrediction<>>(input, output) << std::endl;
        }

        if (stop)
            break;
    }

    std::cout << "Final: " << network.get_loss<simple_nn::NegativeLogGaussianPrediction<>>(input, output) << std::endl;

    // write the predicted data in a file (e.g. to be plotted)
    std::ofstream ofs("nn.dat");
    for (int i = 0; i < 500; ++i) {
        Eigen::VectorXd query(1);
        query << (i / 500.) * 20. - 10.;

        Eigen::MatrixXd out = network.forward(query);

        double log_sigma = out(1, 0);
        double max_logvar = simple_nn::NLGPDefaultParams::max_logvar;
        double min_logvar = simple_nn::NLGPDefaultParams::min_logvar;
        log_sigma = max_logvar - std::log(std::exp(max_logvar - log_sigma) + 1.);
        log_sigma = min_logvar + std::log(std::exp(log_sigma - min_logvar) + 1.);
        double sigma = std::exp(log_sigma);

        ofs << query.transpose() << " " << out(0, 0) << " " << sqrt(sigma) << std::endl;
    }

    std::ofstream ofs_data("data.dat");
    for (int i = 0; i < input.cols(); i++) {
        ofs_data << input.col(i).transpose() << " " << output.col(i).transpose() << std::endl;
    }

    return 0;
}