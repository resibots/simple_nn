#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_loss

#include <ctime>

#include <boost/test/unit_test.hpp>

#include <simple_nn/activation.hpp>

// Check gradient via finite differences method
template <typename Activation>
std::tuple<double, Eigen::MatrixXd, Eigen::MatrixXd> check_grad(const Eigen::MatrixXd& x, double e = 1e-4)
{
    Eigen::MatrixXd finite_diff_result = Eigen::MatrixXd::Zero(x.rows(), x.cols());
    Eigen::MatrixXd analytic_result = Activation::df(x);

    for (int j = 0; j < finite_diff_result.cols(); j++) {
        for (int i = 0; i < finite_diff_result.rows(); i++) {
            Eigen::MatrixXd test1 = x, test2 = x;
            test1(i, j) -= e;
            test2(i, j) += e;
            double r1 = Activation::f(test1)(i, j);
            double r2 = Activation::f(test2)(i, j);

            finite_diff_result(i, j) = (r2 - r1) / (2.0 * e);
        }
    }

    return std::make_tuple((analytic_result - finite_diff_result).norm(), analytic_result, finite_diff_result);
}

BOOST_AUTO_TEST_CASE(test_gradients)
{
    std::srand(std::time(NULL));

    int N = 50;
    int fails = 0;

    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 20).array() * 10.;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::Sigmoid>(input);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 20).array() * 10.;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::Gaussian>(input);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 20).array() * 10.;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::Swish>(input);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 20).array() * 10.;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::Tanh>(input);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 20).array() * 10.;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::ReLU>(input);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);
}