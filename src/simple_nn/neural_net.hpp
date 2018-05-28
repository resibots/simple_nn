#ifndef SIMPLE_NN_NEURAL_NET_HPP
#define SIMPLE_NN_NEURAL_NET_HPP

#include <Eigen/Core>
// #include <Eigen/Dense>

#include <memory>
#include <tuple>

// #include <iostream>

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

    struct NeuralNet {
        void add_layer(std::unique_ptr<Layer>& layer)
        {
            if (_layers.size() > 0) {
                size_t index_prev = _layers.size() - 1;

                assert(_layers[index_prev]->_output == layer->_input);
            }

            _layers.push_back(std::move(layer));
        }

        size_t num_weights() const
        {
            size_t n = 0;
            for (size_t i = 0; i < _layers.size(); i++) {
                n += _layers[i]->num_weights();
            }

            return n;
        }

        Eigen::VectorXd weights() const
        {
            Eigen::VectorXd weights = Eigen::VectorXd::Zero(num_weights());

            size_t offset = 0;
            for (size_t i = 0; i < _layers.size(); i++) {
                Eigen::VectorXd w = _layers[i]->weights_vector();
                weights.segment(offset, w.size()) = w;
                offset += w.size();
            }

            return weights;
        }

        void set_weights(const Eigen::VectorXd& w)
        {
            size_t offset = 0;
            for (size_t i = 0; i < _layers.size(); i++) {
                size_t w_size = _layers[i]->num_weights();
                _layers[i]->set_weights_vector(w.segment(offset, w_size));
                offset += w_size;
            }
        }

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) const
        {
            Eigen::MatrixXd result = input;
            for (size_t i = 0; i < _layers.size(); i++) {
                result = _layers[i]->forward(result);
            }

            return result;
        }

        Eigen::VectorXd backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& output) const
        {
            Eigen::VectorXd gradients = Eigen::VectorXd::Zero(num_weights());

            std::vector<Eigen::MatrixXd> results(_layers.size());

            Eigen::MatrixXd result = input;
            for (size_t i = 0; i < _layers.size(); i++) {
                // std::cout << i << ": " << result.rows() << "x" << result.cols() << std::endl;
                results[i] = result;
                result = _layers[i]->forward(result);
                // results[i] = result;
                // std::cout << result.rows() << "x" << result.cols() << std::endl;
            }

            std::vector<Eigen::MatrixXd> deltas(_layers.size() + 1);
            std::vector<Eigen::MatrixXd> grads(_layers.size());

            deltas.back() = (result.array() - output.array());

            // std::cout << deltas.back().rows() << "x" << deltas.back().cols() << std::endl;

            // std::cout << "error: " << deltas.back() << std::endl;

            size_t k = 0;

            for (int i = _layers.size() - 1; i >= 0; i--) {
                Eigen::MatrixXd delta;
                Eigen::MatrixXd next_delta;
                std::tie(next_delta, delta) = _layers[i]->backward(results[i], deltas[i + 1]);

                Eigen::MatrixXd input_bias = results[i];
                input_bias.conservativeResize(input_bias.rows() + 1, input_bias.cols());
                input_bias.row(input_bias.rows() - 1) = Eigen::VectorXd::Ones(results[i].cols());
                // std::cout << i << ": " << results[i] << std::endl;

                // std::cout << "delta: " << delta << std::endl;
                // std::cout << "input: " << input_bias.transpose() << std::endl;

                grads[k] = delta * input_bias.transpose();
                // std::cout << "grad: " << grads[k] << std::endl;
                // std::cout << grads[k].rows() << "x" << grads[k].cols() << std::endl;
                k++;

                next_delta.conservativeResize(next_delta.rows() - 1, next_delta.cols());
                deltas[i] = next_delta;
            }

            int offset = 0;
            for (int i = grads.size() - 1; i >= 0; i--) {
                int rows = grads[i].rows();
                int cols = grads[i].cols();
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        gradients(offset + r * cols + c) = grads[i](r, c);
                    }
                }

                offset += rows * cols;
            }

            return gradients;
        }

        std::vector<std::unique_ptr<Layer>> _layers;
    };

} // namespace nn

#endif