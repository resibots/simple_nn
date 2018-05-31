#ifndef SIMPLE_NN_LOSS_HPP
#define SIMPLE_NN_LOSS_HPP

#include <Eigen/Core>

namespace simple_nn {
    struct MeanSquaredError {
    public:
        static double f(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return (y.array() - y_d.array()).square().rowwise().sum().sum() / static_cast<double>(y.cols());
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return 2. * (y.array() - y_d.array()) / static_cast<double>(y.cols());
        }
    };

    struct SquaredError {
    public:
        static double f(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return (y.array() - y_d.array()).square().rowwise().sum().sum();
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return 2. * (y.array() - y_d.array());
        }
    };

    struct AbsoluteError {
    public:
        static double f(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return (y.array() - y_d.array()).abs().rowwise().sum().sum() / static_cast<double>(y.cols());
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            Eigen::MatrixXd z = Eigen::MatrixXd::Zero(y.rows(), y.cols());
            Eigen::MatrixXd o = Eigen::MatrixXd::Ones(y.rows(), y.cols()).array() / static_cast<double>(y.cols());

            Eigen::MatrixXd output = Eigen::MatrixXd::Zero(y.rows(), y.cols());
            Eigen::MatrixXd diff = (y.array() - y_d.array());
            output = (diff.array() > 0).select(o, z);
            output = (diff.array() < 0).select(-o, output);

            return output;
        }
    };

    struct CrossEntropyMultiClass {
    public:
        static double f(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return (-y_d.array() * y.array().log()).rowwise().sum().sum();
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return -y_d.array() / y.array();
        }
    };

    struct CrossEntropy {
    public:
        static double f(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return (-y_d.array() * y.array().log() - (1. - y_d.array()) * (1. - y.array()).log()).rowwise().sum().sum();
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return (y.array() - y_d.array()) / (y.array() * (1. - y.array()));
        }
    };
} // namespace simple_nn

#endif