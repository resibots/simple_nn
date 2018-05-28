#ifndef SIMPLE_NN_LAYER_HPP
#define SIMPLE_NN_LAYER_HPP

#include <Eigen/Core>

#include <tuple>

namespace simple_nn {
    struct Layer {
        Layer(size_t input, size_t output) : _input(input), _output(output) {}

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd&) const = 0;

        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const = 0;

        virtual size_t num_weights() const = 0;

        virtual Eigen::VectorXd weights_vector() const = 0;
        virtual void set_weights_vector(const Eigen::VectorXd& w) = 0;

        size_t _input, _output;
    };

    struct FullyConnectedLayer : public Layer {
        FullyConnectedLayer(size_t input, size_t output) : Layer(input, output)
        {
            _W.resize(_output, _input + 1);
        }

        virtual size_t num_weights() const
        {
            return _output * (_input + 1);
        }

        void set_weights(const Eigen::MatrixXd& w)
        {
            assert(w.rows() == static_cast<int>(_output));
            assert(w.cols() == static_cast<int>(_input + 1));
            _W = w;
        }

        virtual void set_weights_vector(const Eigen::VectorXd& w)
        {
            assert(w.size() == static_cast<int>(_output * (_input + 1)));
            _W.resize(_output, _input + 1);

            for (size_t i = 0; i < _output; i++) {
                for (size_t j = 0; j < (_input + 1); j++) {
                    _W(i, j) = w(i * (_input + 1) + j);
                }
            }
        }

        Eigen::MatrixXd weights() const
        {
            return _W;
        }

        virtual Eigen::VectorXd weights_vector() const
        {
            Eigen::VectorXd w(_W.rows() * _W.cols());

            for (size_t i = 0; i < _output; i++) {
                for (size_t j = 0; j < (_input + 1); j++) {
                    w(i * (_input + 1) + j) = _W(i, j);
                }
            }

            return w;
        }

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) const override
        {
            Eigen::MatrixXd input_bias = input;
            input_bias.conservativeResize(input_bias.rows() + 1, input_bias.cols());
            input_bias.row(input_bias.rows() - 1) = Eigen::VectorXd::Ones(input.cols());
            return _W * input_bias;
        }

        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd&, const Eigen::MatrixXd& delta) const override
        {
            return std::make_tuple(_W.transpose() * delta, delta);
        }

        Eigen::MatrixXd _W;
    };

    struct SigmoidLayer : public FullyConnectedLayer {
        SigmoidLayer(size_t input, size_t output) : FullyConnectedLayer(input, output) {}

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) const override
        {
            Eigen::MatrixXd output = FullyConnectedLayer::forward(input);
            return 1. / (1. + (-output).array().exp());
        }

        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& delta) const override
        {
            Eigen::MatrixXd val = forward(input);
            Eigen::MatrixXd grad = (val.array() * (1. - val.array()));
            Eigen::MatrixXd tmp = delta.array() * grad.array();

            return std::make_tuple(_W.transpose() * tmp, tmp);
        }
    };
} // namespace simple_nn

#endif