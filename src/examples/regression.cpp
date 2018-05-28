#include <ctime>
#include <iostream>

#include <simple_nn/neural_net.hpp>

// #include <limbo/tools/random_generator.hpp>

// Check gradient via finite differences method
template <typename NeuralNet>
std::tuple<double, Eigen::VectorXd, Eigen::VectorXd> check_grad(const NeuralNet& net, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::VectorXd& theta, double e = 1e-4)
{
    Eigen::VectorXd analytic_result, finite_diff_result;
    NeuralNet test_net = net;
    test_net.set_weights(theta);

    analytic_result = test_net.backward(x, y);

    finite_diff_result = Eigen::VectorXd::Zero(theta.size());
    for (int j = 0; j < theta.size(); j++) {
        Eigen::VectorXd test1 = theta, test2 = theta;
        test1[j] -= e;
        test2[j] += e;
        test_net.set_weights(test1);
        Eigen::MatrixXd res1 = test_net.forward(x);
        double r1 = (res1.array() - y.array()).square().colwise().sum().sum() / 2.0;
        test_net.set_weights(test2);
        Eigen::MatrixXd res2 = test_net.forward(x);
        double r2 = (res2.array() - y.array()).square().colwise().sum().sum() / 2.0;

        // std::cout << res1 << std::endl
        //           << std::endl;
        // std::cout << "----------------" << std::endl;
        // std::cout << res2 << std::endl
        //           << std::endl;
        // std::cout << "----------------" << std::endl;
        // std::cout << y << std::endl
        //           << std::endl;
        // std::cout << "================" << std::endl;
        // std::cout << r1 << " vs " << r2 << std::endl;

        finite_diff_result[j] = (r2 - r1) / (2.0 * e);
    }

    return std::make_tuple((analytic_result - finite_diff_result).norm(), analytic_result, finite_diff_result);
}

int main()
{
    std::srand(std::time(NULL));

    simple_nn::NeuralNet network;

    // std::unique_ptr<simple_nn::Layer> sigmoid_layer = std::unique_ptr<simple_nn::SigmoidLayer>(new simple_nn::SigmoidLayer(5, 20));
    // std::unique_ptr<simple_nn::Layer> hidden_layer = std::unique_ptr<simple_nn::FullyConnectedLayer>(new simple_nn::FullyConnectedLayer(20, 2));
    // std::unique_ptr<simple_nn::Layer> sigmoid_layer2 = std::unique_ptr<simple_nn::SigmoidLayer>(new simple_nn::SigmoidLayer(2, 20));
    // std::unique_ptr<simple_nn::Layer> sigmoid_layer3 = std::unique_ptr<simple_nn::SigmoidLayer>(new simple_nn::SigmoidLayer(20, 2));

    // network.add_layer(sigmoid_layer);
    // network.add_layer(hidden_layer);
    // network.add_layer(sigmoid_layer2);
    // network.add_layer(sigmoid_layer3);
    network.add_layer<simple_nn::SigmoidLayer>(5, 20);
    network.add_layer<simple_nn::FullyConnectedLayer>(20, 2);
    network.add_layer<simple_nn::SigmoidLayer>(2, 20);
    network.add_layer<simple_nn::SigmoidLayer>(20, 2);

    for (int i = 0; i < 10; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(5, 20).array() * 10.;
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(2, 20); //input.array().cos(); //Eigen::MatrixXd::Random(1, 1);
        // output(0, 0) = std::cos(input(0, 0));

        // std::cout << output << std::endl;
        // std::cout << input << " --> " << output << std::endl;

        Eigen::VectorXd theta = Eigen::VectorXd::Random(network.num_weights()); //limbo::tools::random_vector(network.num_weights()).array() * 2. - 1.;
        // std::cout << theta.transpose() << std::endl;

        network.set_weights(theta);

        // std::cout << "analytic: " << network.backward(input, output).transpose() << std::endl;

        // Eigen::MatrixXd out = network.forward(input);
        // std::cout << "out: " << out << std::endl;

        // Eigen::MatrixXd error = out.array() - output.array();
        // std::cout << "simple: " << error.array() * input.array() << " " << error << std::endl;

        double err;
        Eigen::VectorXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad(network, input, output, theta);

        std::cout << err << std::endl;
        // std::cout << analytic.transpose() << std::endl;
        // std::cout << finite_diff.transpose() << std::endl;
        std::cout << "-----------------" << std::endl;
    }

    // std::cout << grads.size() << std::endl;

    return 0;
}