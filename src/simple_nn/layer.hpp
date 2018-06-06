#ifndef SIMPLE_NN_LAYER_HPP
#define SIMPLE_NN_LAYER_HPP

#include <Eigen/Core>

#include <tuple>

#include <simple_nn/activation.hpp>

namespace simple_nn {
    struct Layer {
    public:
        Layer(size_t input, size_t output) : _input(input), _output(output) {}

        size_t input() const { return _input; }
        size_t output() const { return _output; }

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd&) const = 0;

        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const = 0;

        virtual size_t num_weights() const = 0;

        virtual Eigen::VectorXd weights_vector() const = 0;
        virtual void set_weights_vector(const Eigen::VectorXd& w) = 0;

        virtual std::shared_ptr<Layer> clone() const = 0;

    protected:
        size_t _input, _output;
    };

    template <typename Activation = Linear>
    struct FullyConnectedLayer : public Layer {
    public:
        FullyConnectedLayer(size_t input, size_t output) : Layer(input, output)
        {
            _W.resize(_output, _input + 1);
        }

        virtual std::shared_ptr<Layer> clone() const
        {
            std::shared_ptr<Layer> layer = std::make_shared<FullyConnectedLayer<Activation>>(_input, _output);
            std::static_pointer_cast<FullyConnectedLayer<Activation>>(layer)->_W = _W;

            return layer;
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

        Eigen::MatrixXd compute(const Eigen::MatrixXd& input) const
        {
            Eigen::MatrixXd input_bias = input;
            input_bias.conservativeResize(input_bias.rows() + 1, input_bias.cols());
            input_bias.row(input_bias.rows() - 1) = Eigen::VectorXd::Ones(input.cols());

            return _W * input_bias;
        }

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) const override
        {
            return Activation::f(compute(input));
        }

        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& delta) const override
        {
            Eigen::MatrixXd val = compute(input);
            Eigen::MatrixXd tmp = delta.array() * Activation::df(val).array();
            return std::make_tuple(_W.transpose() * tmp, tmp);
        }

    protected:
        Eigen::MatrixXd _W;
    };
} // namespace simple_nn

#endif