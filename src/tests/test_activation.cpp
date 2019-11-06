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
#define BOOST_TEST_MODULE test_activation

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
            Eigen::MatrixXd t1 = Activation::f(test1);
            Eigen::MatrixXd t2 = Activation::f(test2);
            int index = i;
            if (index >= t1.rows() || index >= t2.rows())
                index = std::min(t1.rows(), t2.rows()) - 1;
            double r1 = t1(index, j);
            double r2 = t2(index, j);

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

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::Softmax>(input);

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

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 20).array() * 10.;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::Cos>(input);

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

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::Sin>(input);

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

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::Multiply>(input);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(2, 20).array() * 10.;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::Divide>(input, 1e-5);

        if (err > 1e-3) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);
}