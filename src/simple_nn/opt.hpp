#ifndef SIMPLE_NN_OPT_HPP
#define SIMPLE_NN_OPT_HPP

#include <tuple>

#include <Eigen/Core>

namespace simple_nn {
    /// Adam optimizer
    /// Equations from: http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms
    /// (I changed a bit the notation; η to α)
    class Adam {
    public:
        struct State {
            void init(const Eigen::VectorXd& init, double b1, double b2)
            {
                b1_n = b1;
                b2_n = b2;
                params = init;
                m = Eigen::VectorXd::Zero(init.size());
                v = Eigen::VectorXd::Zero(init.size());
                val = 0.;
            }

            double b1_n;
            double b2_n;
            double val;
            Eigen::VectorXd params;
            Eigen::VectorXd m, v;
        };

        Adam(double alpha = 0.001, double b1 = 0.9, double b2 = 0.999, double epsilon = 1e-8, double eps_stop = 0.)
            : _alpha(alpha), _b1(b1), _b2(b2), _epsilon(epsilon), _eps_stop(eps_stop) {}

        void set_alpha(double alpha) { _alpha = alpha; }
        void set_beta1(double b1) { _b1 = b1; }
        void set_beta2(double b2) { _b2 = b2; }
        void set_epsilon(double eps) { _epsilon = eps; }
        void set_eps_stop(double eps) { _eps_stop = eps; }

        double alpha() const { return _alpha; }
        double beta1() const { return _b1; }
        double beta2() const { return _b2; }
        double epsilon() const { return _epsilon; }
        double eps_stop() const { return _eps_stop; }

        State state() const { return _state; }

        void reset(const Eigen::VectorXd& init)
        {
            _state.init(init, _b1, _b2);
        }

        template <typename F>
        std::tuple<bool, double, Eigen::VectorXd> optimize_once(const F& f)
        {
            return optimize_once(_state, f);
        }

        template <typename F>
        std::tuple<bool, double, Eigen::VectorXd> optimize_once(State& adam_state, const F& f) const
        {
            assert(_b1 >= 0. && _b1 < 1.);
            assert(_b2 >= 0. && _b2 < 1.);
            assert(_alpha >= 0.);
            assert(adam_state.params.size());
            assert(adam_state.m.size());
            assert(adam_state.v.size());

            Eigen::VectorXd grad;
            Eigen::VectorXd prev_params = adam_state.params;
            std::tie(adam_state.val, grad) = f(adam_state.params);

            adam_state.m.array() = _b1 * adam_state.m.array() + (1. - _b1) * grad.array();
            adam_state.v.array() = _b2 * adam_state.v.array() + (1. - _b2) * grad.array().square();

            double lr = _alpha * std::sqrt(1. - adam_state.b2_n) / (1. - adam_state.b1_n);

            adam_state.params.array() -= lr * adam_state.m.array() / (adam_state.v.array().sqrt() + _epsilon);

            adam_state.b1_n *= _b1;
            adam_state.b2_n *= _b2;

            return {((prev_params - adam_state.params).norm() < _eps_stop), adam_state.val, adam_state.params};
        }

        template <typename F>
        std::tuple<double, Eigen::VectorXd> optimize(const F& f, int max_iterations, const Eigen::VectorXd& init) const
        {
            State adam_state;
            adam_state.init(init, _b1, _b2);

            for (int i = 0; i < max_iterations; ++i) {
                bool stop = false;
                std::tie(stop, std::ignore, std::ignore) = optimize_once(adam_state, f);

                if (stop)
                    break;
            }

            return {adam_state.val, adam_state.params};
        }

    protected:
        double _alpha;
        double _b1;
        double _b2;
        double _epsilon;
        double _eps_stop;

        State _state;
    };

    /// Gradient Ascent with or without momentum (Nesterov or simple)
    /// Equations from: http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms
    /// (I changed a bit the notation; η to α)
    class GradientDescent {
    public:
        struct State {
            void init(const Eigen::VectorXd& init)
            {
                params = init;
                v = Eigen::VectorXd::Zero(init.size());
                val = 0.;
            }

            double val;
            Eigen::VectorXd params;
            Eigen::VectorXd v;
        };

        GradientDescent(double alpha = 0.001, double gamma = 0., bool nesterov = false, double eps_stop = 0.)
            : _alpha(alpha), _gamma(gamma), _nesterov(nesterov), _eps_stop(eps_stop) {}

        void set_alpha(double alpha) { _alpha = alpha; }
        void set_gamma(double gamma) { _gamma = gamma; }
        void set_nesterov(bool nesterov) { _nesterov = nesterov; }
        void set_eps_stop(double eps) { _eps_stop = eps; }

        double alpha(double alpha) { return _alpha; }
        double gamma(double gamma) { return _gamma; }
        double nesterov(bool nesterov) { return _nesterov; }
        double eps_stop(double eps) { return _eps_stop; }

        State state() const { return _state; }

        void reset(const Eigen::VectorXd& init)
        {
            _state.init(init);
        }

        template <typename F>
        std::tuple<bool, double, Eigen::VectorXd> optimize_once(const F& f)
        {
            return optimize_once(_state, f);
        }

        template <typename F>
        std::tuple<bool, double, Eigen::VectorXd> optimize_once(State& sga_state, const F& f) const
        {
            assert(_gamma >= 0. && _gamma < 1.);
            assert(_alpha >= 0.);
            assert(sga_state.params.size());
            assert(sga_state.v.size());

            Eigen::VectorXd grad;

            Eigen::VectorXd prev_params = sga_state.params;
            Eigen::VectorXd query_params = sga_state.params;
            // if Nesterov momentum, change query parameters
            if (_nesterov) {
                query_params.array() += _gamma * sga_state.v.array();
            }
            std::tie(sga_state.val, grad) = f(query_params);

            sga_state.v = _gamma * sga_state.v.array() + _alpha * grad.array();

            sga_state.params.array() -= sga_state.v.array();

            return {((prev_params - sga_state.params).norm() < _eps_stop), sga_state.val, sga_state.params};
        }

        template <typename F>
        std::tuple<double, Eigen::VectorXd> optimize(const F& f, int max_iterations, const Eigen::VectorXd& init) const
        {
            State sga_state;
            sga_state.init(init);

            for (int i = 0; i < max_iterations; ++i) {
                bool stop = false;
                std::tie(stop, std::ignore, std::ignore) = optimize_once(sga_state, f);

                if (stop)
                    break;
            }

            return {sga_state.val, sga_state.params};
        }

    protected:
        double _alpha;
        double _gamma;
        bool _nesterov;
        double _eps_stop;

        State _state;
    };
} // namespace simple_nn

#endif