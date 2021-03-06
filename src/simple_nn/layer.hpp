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
#ifndef SIMPLE_NN_LAYER_HPP
#define SIMPLE_NN_LAYER_HPP

#include <Eigen/Core>

#include <memory>
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

    template <typename LayerType>
    struct ScaledLayer : public LayerType {
    public:
        template <typename... Args>
        ScaledLayer(double scaling, Args&&... args) : LayerType(std::forward<Args>(args)...), _scaling(Eigen::VectorXd::Constant(LayerType::output(), scaling)) {}
        template <typename... Args>
        ScaledLayer(const Eigen::VectorXd& scaling, Args&&... args) : LayerType(std::forward<Args>(args)...), _scaling(scaling) {}

        ScaledLayer(double scaling, const LayerType& other) : LayerType(other), _scaling(Eigen::VectorXd::Constant(LayerType::output(), scaling)) {}
        ScaledLayer(const Eigen::VectorXd& scaling, const LayerType& other) : LayerType(other), _scaling(scaling)
        {
            assert(static_cast<size_t>(_scaling.size()) == LayerType::output());
        }

        virtual std::shared_ptr<Layer> clone() const
        {
            std::shared_ptr<Layer> layer = std::make_shared<ScaledLayer<LayerType>>(_scaling, *std::static_pointer_cast<LayerType>(LayerType::clone()));

            return layer;
        }

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) const override
        {
            Eigen::MatrixXd result = LayerType::forward(input);
            result.array().colwise() *= _scaling.array();

            return result;
        }

        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& delta) const override
        {
            Eigen::MatrixXd d = delta;
            d.array().colwise() *= _scaling.array();
            std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> result = LayerType::backward(input, d);

            return result;
        }

    protected:
        Eigen::VectorXd _scaling;
    };

    template <typename LayerType>
    struct AdditionLayer : public LayerType {
    public:
        template <typename... Args>
        AdditionLayer(double addition, Args&&... args) : LayerType(std::forward<Args>(args)...), _addition(Eigen::VectorXd::Constant(LayerType::output(), addition)) {}
        template <typename... Args>
        AdditionLayer(const Eigen::VectorXd& addition, Args&&... args) : LayerType(std::forward<Args>(args)...), _addition(addition) {}

        AdditionLayer(double addition, const LayerType& other) : LayerType(other), _addition(Eigen::VectorXd::Constant(LayerType::output(), addition)) {}
        AdditionLayer(const Eigen::VectorXd& addition, const LayerType& other) : LayerType(other), _addition(addition)
        {
            assert(static_cast<size_t>(_addition.size()) == LayerType::output());
        }

        virtual std::shared_ptr<Layer> clone() const
        {
            std::shared_ptr<Layer> layer = std::make_shared<AdditionLayer<LayerType>>(_addition, *std::static_pointer_cast<LayerType>(LayerType::clone()));

            return layer;
        }

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) const override
        {
            Eigen::MatrixXd result = LayerType::forward(input);
            result.array().colwise() += _addition.array();

            return result;
        }

    protected:
        Eigen::VectorXd _addition;
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

    /// CustomLayer allows the implementation of layers with different activations per node
    /// For example, it can allow the implementation of:
    /// Extrapolation and learning equations
    /// by Georg Martius, Christoph H. Lampert, 2016
    /// https://arxiv.org/abs/1610.02995, https://arxiv.org/abs/1806.07259
    /// code: https://github.com/martius-lab/EQL
    struct CustomLayer : public Layer {
    public:
        CustomLayer(size_t input) : Layer(input, 0) { _layer_output = 0; }

        virtual std::shared_ptr<Layer> clone() const
        {
            std::shared_ptr<Layer> layer = std::make_shared<CustomLayer>(_input);
            std::static_pointer_cast<CustomLayer>(layer)->_output = _output;
            std::static_pointer_cast<CustomLayer>(layer)->_layer_output = _layer_output;
            std::static_pointer_cast<CustomLayer>(layer)->_W = _W;
            std::static_pointer_cast<CustomLayer>(layer)->_forward_activations = _forward_activations;
            std::static_pointer_cast<CustomLayer>(layer)->_backward_activations = _backward_activations;
            std::static_pointer_cast<CustomLayer>(layer)->_inputs = _inputs;
            std::static_pointer_cast<CustomLayer>(layer)->_outputs = _outputs;

            return layer;
        }

        virtual size_t num_weights() const
        {
            return _layer_output * (_input + 1);
        }

        void set_weights(const Eigen::MatrixXd& w)
        {
            assert(w.rows() == static_cast<int>(_layer_output));
            assert(w.cols() == static_cast<int>(_input + 1));
            _W = w;
        }

        virtual void set_weights_vector(const Eigen::VectorXd& w)
        {
            assert(w.size() == static_cast<int>(_layer_output * (_input + 1)));
            _W.resize(_layer_output, _input + 1);

            for (size_t i = 0; i < _layer_output; i++) {
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

            for (size_t i = 0; i < _layer_output; i++) {
                for (size_t j = 0; j < (_input + 1); j++) {
                    w(i * (_input + 1) + j) = _W(i, j);
                }
            }

            return w;
        }

        size_t layer_output() const { return _layer_output; }

        Eigen::MatrixXd compute(const Eigen::MatrixXd& input) const
        {
            Eigen::MatrixXd input_bias = input;
            input_bias.conservativeResize(input_bias.rows() + 1, input_bias.cols());
            input_bias.row(input_bias.rows() - 1) = Eigen::VectorXd::Ones(input.cols());

            return _W * input_bias;
        }

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) const override
        {
            Eigen::MatrixXd tmp = compute(input);
            Eigen::MatrixXd out(_output, input.cols());
            size_t total_r = 0, total_o = 0;
            for (size_t i = 0; i < _inputs.size(); i++) {
                size_t r = _inputs[i];
                size_t o = _outputs[i];
                out.block(total_o, 0, o, input.cols()) = _forward_activations[i](tmp.block(total_r, 0, r, input.cols()));
                total_r += r;
                total_o += o;
            }

            return out;
        }

        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& delta) const override
        {
            Eigen::MatrixXd output = compute(input);
            Eigen::MatrixXd dlt = delta;

            size_t total_r = 0;
            for (size_t i = 0; i < _inputs.size(); i++) {
                size_t r = _inputs[i];
                size_t o = _outputs[i];
                output.block(total_r, 0, r, input.cols()) = _backward_activations[i](output.block(total_r, 0, r, input.cols()));
                if (r > o) {
                    int n = r - o;
                    int N = dlt.rows();
                    dlt.conservativeResize(N + n, dlt.cols());
                    int moving = N - total_r - 1;
                    dlt.block(N - moving - 1, 0, moving, dlt.cols()) = dlt.block(total_r + 1, 0, moving, dlt.cols());
                    for (int k = 0; k < n; k++) {
                        dlt.row(total_r + 1 + k) = dlt.row(total_r);
                    }
                }
                total_r += r;
            }

            Eigen::MatrixXd tmp = dlt.array() * output.array();
            return std::make_tuple(_W.transpose() * tmp, tmp);
        }

        template <typename Activation>
        void add_activation(size_t dim_in, size_t dim_out = 0)
        {
            size_t out = (dim_out > 0) ? dim_out : dim_in;
            _layer_output += dim_in;
            _output += out;

            _inputs.push_back(dim_in);
            _outputs.push_back(out);

            _forward_activations.push_back(Activation::f);
            _backward_activations.push_back(Activation::df);

            _W.conservativeResize(_layer_output, _input);
        }

    protected:
        Eigen::MatrixXd _W;

        std::vector<std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>> _forward_activations;
        std::vector<std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>> _backward_activations;
        std::vector<size_t> _inputs, _outputs;
        size_t _layer_output;
    };
} // namespace simple_nn

#endif