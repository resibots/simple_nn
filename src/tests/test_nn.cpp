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

    network.add_layer<simple_nn::SigmoidLayer>(5, 20);
    network.add_layer<simple_nn::FullyConnectedLayer>(20, 2);
    network.add_layer<simple_nn::TanhLayer>(2, 20);
    network.add_layer<simple_nn::SigmoidLayer>(20, 2);

    int N = 50;
    int fails = 0;

    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(5, 20).array() * 10.;
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(2, 20);

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

    fails = 0;

    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(5, 20).array() * 10.;
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(2, 20);

        Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights());

        network.set_weights(theta);

        double err;
        Eigen::VectorXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::SquaredError>(network, input, output, theta);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;

    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(5, 20).array() * 10.;
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(2, 20);

        Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights());

        network.set_weights(theta);

        double err;
        Eigen::VectorXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::AbsoluteError>(network, input, output, theta);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;

    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(5, 20).array() * 10.;
        Eigen::MatrixXd output = (Eigen::MatrixXd::Random(2, 20).array() + 1.) / 2.;

        Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights());

        network.set_weights(theta);

        double err;
        Eigen::VectorXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::CrossEntropy>(network, input, output, theta);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;

    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(5, 20).array() * 10.;
        Eigen::MatrixXd output = (Eigen::MatrixXd::Random(2, 20).array() + 1.) / 2.;

        Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights());

        network.set_weights(theta);

        double err;
        Eigen::VectorXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::CrossEntropyMultiClass>(network, input, output, theta);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);
}