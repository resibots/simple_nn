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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_nn

#include <ctime>

#include <boost/test/unit_test.hpp>

#include <simple_nn/loss.hpp>
#include <simple_nn/neural_net.hpp>

// Check gradient via finite differences method
template <typename Loss>
std::tuple<double, Eigen::VectorXd, Eigen::VectorXd> check_grad(const simple_nn::NeuralNet& net, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::VectorXd& theta, double e = 1e-4)
{
    Eigen::VectorXd analytic_result, finite_diff_result;
    simple_nn::NeuralNet test_net = net;
    test_net.set_weights(theta);

    analytic_result = test_net.backward<Loss>(x, y);

    finite_diff_result = Eigen::VectorXd::Zero(theta.size());
    for (int j = 0; j < theta.size(); j++) {
        Eigen::VectorXd test1 = theta, test2 = theta;
        test1[j] -= e;
        test2[j] += e;
        test_net.set_weights(test1);
        Eigen::MatrixXd res1 = test_net.forward(x);
        double r1 = Loss::f(res1, y);
        test_net.set_weights(test2);
        Eigen::MatrixXd res2 = test_net.forward(x);
        double r2 = Loss::f(res2, y);

        finite_diff_result[j] = (r2 - r1) / (2.0 * e);
    }

    return std::make_tuple((analytic_result - finite_diff_result).norm(), analytic_result, finite_diff_result);
}

BOOST_AUTO_TEST_CASE(test_gradients)
{
    std::srand(std::time(NULL));

    simple_nn::NeuralNet network;

    network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Sigmoid>>(5, 20);
    network.add_layer<simple_nn::FullyConnectedLayer<>>(20, 2);
    network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Swish>>(2, 20);
    network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::ReLU>>(20, 20);
    network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(20, 20);
    network.add_layer<simple_nn::AdditionLayer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>>(-1., 20, 20);
    network.add_layer<simple_nn::ScaledLayer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>>(2., 20, 4);

    int N = 50;
    int fails = 0;

    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(5, 20).array() * 10.;
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(4, 20);

        Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights());

        network.set_weights(theta);

        double err;
        Eigen::VectorXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::MeanSquaredError>(network, input, output, theta);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);
}