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
#ifndef SIMPLE_NN_LOSS_HPP
#define SIMPLE_NN_LOSS_HPP

#include <Eigen/Core>

namespace simple_nn {
    struct MeanSquaredError {
    public:
        static double f(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return (y.array() - y_d.array()).square().colwise().sum().sum() / static_cast<double>(y.cols());
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
            return (y.array() - y_d.array()).square().colwise().sum().sum();
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
            return (y.array() - y_d.array()).abs().colwise().sum().sum() / static_cast<double>(y.cols());
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
            return (-y_d.array() * y.array().log()).colwise().sum().sum();
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
            return (-y_d.array() * y.array().log() - (1. - y_d.array()) * (1. - y.array()).log()).colwise().sum().sum();
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            return (y.array() - y_d.array()) / (y.array() * (1. - y.array()));
        }
    };

    struct NLGPDefaultParams {
        static constexpr double max_logvar = 0.5;
        static constexpr double min_logvar = -10;
    };

    // output is gaussian with diagonal covariance
    // mu0, mu1, ... , log(s00), log(s11), ...
    template <typename Params = NLGPDefaultParams>
    struct NegativeLogGaussianPrediction {
    public:
        static constexpr double max_logvar = Params::max_logvar;
        static constexpr double min_logvar = Params::min_logvar;

        static double f(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            size_t dim = y.rows() / 2;

            Eigen::MatrixXd mu = y.block(0, 0, dim, y.cols());
            Eigen::MatrixXd log_sigma = y.block(dim, 0, dim, y.cols());
            log_sigma = max_logvar - ((max_logvar - log_sigma.array()).exp() + 1.).log();
            log_sigma = min_logvar + ((log_sigma.array() - min_logvar).exp() + 1.).log();

            Eigen::MatrixXd sigma = log_sigma.array().exp();

            Eigen::MatrixXd diff = mu.array() - y_d.array();
            Eigen::MatrixXd inv_sigma = 1. / (sigma.array() + 1e-8);
            Eigen::VectorXd logdet_sigma = log_sigma.colwise().sum();

            double loss = 0.;
            for (int i = 0; i < y.cols(); i++) {
                Eigen::MatrixXd inv_S = inv_sigma.col(i).asDiagonal();

                loss += diff.col(i).transpose() * inv_S * diff.col(i) + logdet_sigma(i);
            }

            return loss;
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_d)
        {
            size_t dim = y.rows() / 2;

            Eigen::MatrixXd mu = y.block(0, 0, dim, y.cols());
            Eigen::MatrixXd log_sigma = y.block(dim, 0, dim, y.cols());
            Eigen::MatrixXd l1 = max_logvar - ((max_logvar - log_sigma.array()).exp() + 1.).log();
            Eigen::MatrixXd l2 = min_logvar + ((l1.array() - min_logvar).exp() + 1.).log();

            Eigen::MatrixXd sigma = l2.array().exp();

            Eigen::MatrixXd diff = mu.array() - y_d.array();
            Eigen::MatrixXd inv_sigma = 1. / (sigma.array() + 1e-8);
            Eigen::VectorXd logdet_sigma = l2.colwise().sum();

            Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(y.rows(), y.cols());

            // gradient for the first term
            grad.block(0, 0, dim, y.cols()) = 2. * diff.array() * inv_sigma.array();
            // grad.block(dim, 0, dim, y.cols()) = -(diff.array() * inv_sigma.array()).square(); // this is for non log/exp

            for (int i = 0; i < y.cols(); i++) {
                Eigen::VectorXd diff_sq = diff.col(i).array().square();
                grad.col(i).tail(dim) = -diff_sq.array() * inv_sigma.col(i).array(); // this is for without bounds for variance
                grad.col(i).tail(dim) = -diff_sq.array() * inv_sigma.col(i).array() * logistic(l1.col(i).array() - min_logvar).array() * logistic(max_logvar - log_sigma.col(i).array()).array();
            }

            // gradient for log-det sigma
            for (int i = 0; i < y.cols(); i++) {
                // grad.col(i).tail(dim).array() += Eigen::VectorXd::Ones(dim).array(); // this is for without bounds for variance
                grad.col(i).tail(dim).array() += logistic(l1.col(i).array() - min_logvar).array() * logistic(max_logvar - log_sigma.col(i).array()).array();
            }

            return grad;
        }

        static Eigen::VectorXd logistic(const Eigen::VectorXd& input)
        {
            return 1. / (1. + (-input).array().exp());
        }
    };
} // namespace simple_nn

#endif