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

    // output is gaussian with diagonal covariance
    // mu0, mu1, ... , log(s00), log(s11), ...
    struct NegativeLogGaussianPrediction {
    public:
        static double f(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            size_t dim = y.rows() / 2;
            Eigen::MatrixXd mu = y.block(0, 0, dim, y.cols());
            Eigen::MatrixXd sigma = y.block(dim, 0, dim, y.cols()).array().exp();

            Eigen::MatrixXd diff = mu.array() - y_d.array();
            Eigen::MatrixXd inv_sigma = 1. / sigma.array();
            Eigen::VectorXd logdet_sigma = sigma.colwise().sum();

            double loss = 0.;
            for (int i = 0; i < y.cols(); i++) {
                Eigen::MatrixXd inv_S(dim, dim);
                inv_S.diagonal() = inv_sigma.col(i);

                loss += diff.col(i).transpose() * inv_S * diff.col(i) + logdet_sigma(i);
            }

            return loss;
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            size_t dim = y.rows() / 2;
            Eigen::MatrixXd mu = y.block(0, 0, dim, y.cols());
            Eigen::MatrixXd sigma = y.block(dim, 0, dim, y.cols()).array().exp();

            Eigen::MatrixXd diff = mu.array() - y_d.array();
            Eigen::MatrixXd inv_sigma = 1. / sigma.array();
            Eigen::VectorXd logdet_sigma = sigma.colwise().sum();

            Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(y.rows(), y.cols());

            // gradient for the first term
            grad.block(0, 0, dim, y.cols()) = 2. * diff.array() * inv_sigma.array();

            // grad.block(dim, 0, dim, y.cols()) = -(diff.array() * inv_sigma.array()).square(); // this is for non log/exp
            for (int i = 0; i < y.cols(); i++) {
                Eigen::MatrixXd inv_S(dim, dim);
                inv_S.diagonal() = inv_sigma.col(i);

                grad.col(i).tail(dim) = -(diff.col(i).transpose() * inv_S * diff.col(i));
            }

            // gradient for log-det sigma
            for (int i = 0; i < y.cols(); i++) {
                grad.col(i).tail(dim).array() += logdet_sigma(i); // Eigen::VectorXd::Ones(dim); // this is for non log/exp
            }

            return grad;
        }
    };
} // namespace simple_nn

#endif