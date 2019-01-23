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
#define BOOST_TEST_MODULE test_loss

#include <ctime>

#include <boost/test/unit_test.hpp>

#include <simple_nn/loss.hpp>

// Check gradient via finite differences method
template <typename Loss>
std::tuple<double, Eigen::MatrixXd, Eigen::MatrixXd> check_grad(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double e = 1e-4)
{
    Eigen::MatrixXd finite_diff_result = Eigen::MatrixXd::Zero(x.rows(), x.cols());
    Eigen::MatrixXd analytic_result = Loss::df(x, y);

    for (int j = 0; j < finite_diff_result.cols(); j++) {
        for (int i = 0; i < finite_diff_result.rows(); i++) {
            Eigen::MatrixXd test1 = x, test2 = x;
            test1(i, j) -= e;
            test2(i, j) += e;
            double r1 = Loss::f(test1, y);
            double r2 = Loss::f(test2, y);

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
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(4, 20).array() * 10.;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::MeanSquaredError>(input, output);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 20).array() * 10.;
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(4, 20).array() * 10.;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::SquaredError>(input, output);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 20).array() * 10.;
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(4, 20).array() * 10.;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::AbsoluteError>(input, output);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(6, 20);
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(3, 20);

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::NegativeLogGaussianPrediction<>>(input, output, 1e-8);
        // std::cout << analytic << std::endl
        //           << std::endl
        //           << finite_diff << std::endl;
        // std::cout << "Error: " << err << std::endl
        //           << std::endl;

        if (err > 1e-4) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 20).array() * 0.5 + 0.5;
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(4, 20).array() * 0.5 + 0.5;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::CrossEntropyMultiClass>(input, output, 1e-8);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);

    fails = 0;
    for (int i = 0; i < N; i++) {
        Eigen::MatrixXd input = Eigen::MatrixXd::Random(4, 20).array() * 0.5 + 0.5;
        Eigen::MatrixXd output = Eigen::MatrixXd::Random(4, 20).array() * 0.5 + 0.5;

        double err;
        Eigen::MatrixXd analytic, finite_diff;

        std::tie(err, analytic, finite_diff) = check_grad<simple_nn::CrossEntropy>(input, output, 1e-8);

        if (err > 1e-5) {
            fails++;
        }
    }

    BOOST_CHECK(fails < N / 3);
}