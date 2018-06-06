#ifndef SIMPLE_NN_ACTIVATION_HPP
#define SIMPLE_NN_ACTIVATION_HPP

#include <Eigen/Core>

namespace simple_nn {

    struct Linear {
        static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
        {
            return input;
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
        {
            return Eigen::MatrixXd::Ones(input.rows(), input.cols());
        }
    };

    struct Sigmoid {
        static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
        {
            return 1. / (1. + (-input).array().exp());
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
        {
            return (input.array() * (1. - input.array()));
        }
    };

    struct Tanh {
        static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
        {
            return input.array().tanh();
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
        {
            return 1. - input.array().square();
        }
    };
} // namespace simple_nn

#endif