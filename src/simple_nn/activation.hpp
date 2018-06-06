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
            Eigen::MatrixXd value = f(input);
            return (value.array() * (1. - value.array()));
        }
    };

    struct Swish {
        static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
        {
            return input.array() * Sigmoid::f(input).array();
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
        {
            Eigen::MatrixXd value = f(input);
            return value.array() + Sigmoid::f(input).array() * (1. - value.array());
        }
    };

    struct Tanh {
        static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
        {
            return input.array().tanh();
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
        {
            Eigen::MatrixXd value = f(input);
            return 1. - value.array().square();
        }
    };

    struct ReLU {
        static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
        {
            Eigen::MatrixXd z = Eigen::MatrixXd::Zero(input.rows(), input.cols());
            Eigen::MatrixXd output = (input.array() > 0).select(input, z);

            return output;
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
        {
            Eigen::MatrixXd z = Eigen::MatrixXd::Zero(input.rows(), input.cols());
            Eigen::MatrixXd o = Eigen::MatrixXd::Ones(input.rows(), input.cols());
            Eigen::MatrixXd output = (input.array() > 0).select(o, z);

            return output;
        }
    };
} // namespace simple_nn

#endif