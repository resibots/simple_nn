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

    /// Extrapolation and learning equations
    /// by Georg Martius, Christoph H. Lampert, 2016
    /// https://arxiv.org/abs/1610.02995
    /// code: https://github.com/martius-lab/EQL
    /// Assume functions: identity, cosine, sine, multiplication (in that order and fixed)
    /// TO-DO: Add division layer: https://arxiv.org/abs/1806.07259
    /// TO-DO: Make the functions generic and arbitrary in number
    struct EquationLayer : public Layer {
    public:
        EquationLayer(size_t input) : Layer(input, 4)
        {
            _W.resize(_output + 1, _input + 1);
        }

        virtual std::shared_ptr<Layer> clone() const
        {
            std::shared_ptr<Layer> layer = std::make_shared<EquationLayer>(_input);
            std::static_pointer_cast<EquationLayer>(layer)->_W = _W;

            return layer;
        }

        virtual size_t num_weights() const
        {
            return (_output + 1) * (_input + 1);
        }

        void set_weights(const Eigen::MatrixXd& w)
        {
            assert(w.rows() == static_cast<int>(_output + 1));
            assert(w.cols() == static_cast<int>(_input + 1));
            _W = w;
        }

        virtual void set_weights_vector(const Eigen::VectorXd& w)
        {
            assert(w.size() == static_cast<int>((_output + 1) * (_input + 1)));
            _W.resize(_output + 1, _input + 1);

            for (size_t i = 0; i < (_output + 1); i++) {
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

            for (size_t i = 0; i < (_output + 1); i++) {
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
            Eigen::MatrixXd output = compute(input);
            Eigen::MatrixXd out(4, output.cols());
            out.row(0) = output.row(0);
            out.row(1) = output.row(1).array().cos();
            out.row(2) = output.row(2).array().sin();
            out.row(3) = output.row(3).array() * output.row(4).array();

            return out;
        }

        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& delta) const override
        {
            Eigen::MatrixXd output = compute(input);
            Eigen::MatrixXd out(5, output.cols());
            out.row(0) = Eigen::VectorXd::Ones(output.cols());
            out.row(1) = -output.row(1).array().sin();
            out.row(2) = output.row(2).array().cos();
            out.row(3) = output.row(4).array();
            out.row(4) = output.row(3).array();

            Eigen::MatrixXd dlt = delta;
            dlt.conservativeResize(dlt.rows() + 1, dlt.cols());
            dlt.row(dlt.rows() - 1) = dlt.row(dlt.rows() - 2);

            Eigen::MatrixXd tmp = dlt.array() * out.array();
            return std::make_tuple(_W.transpose() * tmp, tmp);
        }

    protected:
        Eigen::MatrixXd _W;
    };
} // namespace simple_nn

#endif