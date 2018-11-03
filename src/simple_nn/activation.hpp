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

    struct Softmax {
        static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
        {
            Eigen::MatrixXd tmp = input.array().exp();
            Eigen::VectorXd t = tmp.colwise().sum();
            tmp.array().rowwise() /= t.transpose().array();
            return tmp;
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
        {
            Eigen::MatrixXd value = f(input);
            return (value.array() * (1. - value.array()));
        }
    };

    struct Gaussian {
        static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
        {
            return (-input.array().square()).exp();
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
        {
            Eigen::MatrixXd value = f(input);
            return -2. * input.array() * value.array();
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

    struct Cos {
        static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
        {
            return input.array().cos();
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
        {
            return -input.array().sin();
        }
    };

    struct Sin {
        static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
        {
            return input.array().sin();
        }

        static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
        {
            return input.array().cos();
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