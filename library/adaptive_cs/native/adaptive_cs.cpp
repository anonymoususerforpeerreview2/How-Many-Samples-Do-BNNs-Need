#include "adaptive_cs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if defined(ADAPTIVE_CS_HAS_BOOST)
#include <boost/math/tools/roots.hpp>
#endif

#if defined(ADAPTIVE_CS_HAS_GSL)
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#endif

#ifdef ADAPTIVE_CS_OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Numerical helpers: monotone root finding and stable wealth arithmetic.
// ---------------------------------------------------------------------------

namespace // Anonymous namespace containing helper functions such as: bisection, brent, solve_root
{
    constexpr double kLogExpLimit = 700.0;
    constexpr double kMinimumVariance = 1e-12;

    // These component values are also part of the numerical implementation.
    constexpr int kPlus = 0;
    constexpr int kMinus = 1;

    enum class ParallelMode
    {
        None,
        Batch,
        Classes
    };

    ParallelMode choose_parallel_mode(
        std::size_t n_active,
        std::size_t n_classes,
        bool compute_bounds)
    {
#ifdef ADAPTIVE_CS_OPENMP
        if (n_active < 2 && n_classes < 2)
        {
            return ParallelMode::None;
        }

        const std::size_t thread_count =
            static_cast<std::size_t>(std::max(1, omp_get_max_threads()));

        // Intermediate draws only update tracker state, so the independent
        // active trackers are the only useful parallel work.
        if (!compute_bounds)
        {
            return n_active >= 2 ? ParallelMode::Batch : ParallelMode::None;
        }

        // On the final draw, choose the axis that exposes more useful
        // parallel work. Prefer the batch when it can fill the OpenMP team;
        // otherwise use class-level parallelism when the class dimension can.
        const bool batch_saturates = n_active >= thread_count;
        const bool classes_saturate = n_classes >= thread_count;
        if (n_active >= 2 && (batch_saturates || !classes_saturate))
        {
            return ParallelMode::Batch;
        }
        if (n_classes >= 2 && classes_saturate)
        {
            return ParallelMode::Classes;
        }
        if (n_active >= 2 && n_active >= n_classes)
        {
            return ParallelMode::Batch;
        }
        return ParallelMode::None;
#else
        (void)n_active;
        (void)n_classes;
        (void)compute_bounds;
        return ParallelMode::None;
#endif
    }

    double safe_exp(double log_value)
    {
        return std::exp(std::min(log_value, kLogExpLimit));
    }

    double log_initial_capital(double theta, int component)
    {
        return component == kPlus ? std::log(theta) : std::log1p(-theta);
    }

    // All root solvers in this file use the same orientation: f is assumed to
    // be non-decreasing, with f(lo) <= 0 <= f(hi).  The returned pair is kept
    // in coordinate order and encloses the zero.  Both tolerances are required
    // so that a small residual cannot terminate with a wide bracket.
    void validate_root_bracket(
        double lo, double hi, double flo, double fhi)
    {
        if (!std::isfinite(lo) || !std::isfinite(hi) ||
            !std::isfinite(flo) || !std::isfinite(fhi))
        {
            throw std::invalid_argument("root bracket must be finite");
        }
        if (!(lo < hi))
        {
            throw std::invalid_argument("root bracket requires lo < hi");
        }
        if (flo > 0.0 || fhi < 0.0)
        {
            throw std::invalid_argument(
                "root bracket must satisfy f(lo) <= 0 <= f(hi)");
        }
    }

    bool root_converged(
        double lo, double hi, double flo, double fhi,
        double x_tolerance, double y_tolerance)
    {
        return std::abs(hi - lo) <= x_tolerance &&
               std::min(std::abs(flo), std::abs(fhi)) <= y_tolerance;
    }

    struct RootTracePoint
    {
        double lower;
        double upper;
        std::size_t function_evaluations;
    };

    // Mirrors the sign-changing bracket maintained by TOMS 748.  Boost calls
    // the expensive function once per accepted proposal; observing those
    // cache misses gives the bracket after every re-bracketing step without
    // changing the production solver's interpolation logic.
    class RootTraceRecorder
    {
    public:
        RootTraceRecorder(
            double lo, double hi, double flo, double fhi, bool reverse_coordinate)
            : lo_(lo), hi_(hi), reverse_coordinate_(reverse_coordinate)
        {
            if (flo == 0.0)
            {
                lo_ = hi_ = lo;
            }
            else if (fhi == 0.0)
            {
                lo_ = hi_ = hi;
            }
            record(2);
        }

        void observe(double x, double value, std::size_t evaluations)
        {
            if (value == 0.0)
            {
                lo_ = hi_ = x;
            }
            else if (value <= 0.0)
            {
                lo_ = x;
            }
            else
            {
                hi_ = x;
            }
            record(evaluations);
        }

        const std::vector<RootTracePoint>& points() const noexcept { return points_; }

    private:
        void record(std::size_t evaluations)
        {
            if (reverse_coordinate_)
            {
                points_.push_back({1.0 - hi_, 1.0 - lo_, evaluations});
            }
            else
            {
                points_.push_back({lo_, hi_, evaluations});
            }
        }

        double lo_;
        double hi_;
        bool reverse_coordinate_;
        std::vector<RootTracePoint> points_;
    };

    // Strict midpoint bisection.  We retain the complete sign-changing bracket
    // because the callers use the outside side of that bracket for conservative
    // confidence bounds.
    template <typename Function>
    std::pair<double, double> bisection_root(
        Function &&f,
        double lo,
        double hi,
        double flo,
        double fhi,
        int max_steps,
        double x_tolerance,
        double y_tolerance)
    {
        if (flo == 0.0)
            return {lo, lo};
        if (fhi == 0.0)
            return {hi, hi};

        for (int step = 0; step < max_steps; ++step)
        {
            if (root_converged(lo, hi, flo, fhi, x_tolerance, y_tolerance))
            {
                return {lo, hi};
            }

            const double proposal = 0.5 * (lo + hi);

            const double fproposal = f(proposal);
            if (!std::isfinite(fproposal))
            {
                throw std::runtime_error("root function returned a non-finite value");
            }
            if (fproposal == 0.0)
            {
                return {proposal, proposal};
            }
            if (fproposal <= 0.0)
            {
                lo = proposal;
                flo = fproposal;
            }
            else
            {
                hi = proposal;
                fhi = fproposal;
            }
        }
        if (!root_converged(lo, hi, flo, fhi, x_tolerance, y_tolerance))
        {
            throw std::runtime_error(
                "bisection root solver did not satisfy x/y tolerances");
        }
        return {lo, hi};
    }

    template <typename Function>
    double gsl_function_adapter(double x, void *params)
    {
        using FunctionType = std::remove_reference_t<Function>;
        const double value = (*static_cast<FunctionType *>(params))(x);
        return std::isfinite(value)
                   ? value
                   : std::numeric_limits<double>::quiet_NaN();
    }

    // Brent-Dekker root finding supplied by GNU Scientific Library.  GSL keeps
    // the current bracket explicitly; we apply the same x/y stopping rule as
    // the other solvers and return that bracket rather than its point estimate.
    template <typename Function>
    std::pair<double, double> brent_root(
        Function &&f,
        double lo,
        double hi,
        double flo,
        double fhi,
        int max_steps,
        double x_tolerance,
        double y_tolerance)
    {
    // This function finds a root of f(x) inside [lo, hi] using GSL's Brent solver.

#if defined(ADAPTIVE_CS_HAS_GSL)
        if (flo == 0.0)
            return {lo, lo};
        if (fhi == 0.0)
            return {hi, hi};

        using FunctionType = std::remove_reference_t<Function>;
        FunctionType &function = f;
        gsl_function gsl_f;
        gsl_f.function = &gsl_function_adapter<FunctionType>;
        gsl_f.params = const_cast<void *>(static_cast<const void *>(&function));

        gsl_root_fsolver *solver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
        if (solver == nullptr)
        {
            throw std::runtime_error("could not allocate GSL Brent solver");
        }

        struct SolverGuard
        {
            gsl_root_fsolver *solver;
            ~SolverGuard() { gsl_root_fsolver_free(solver); }
        } guard{solver};

        const int set_status = gsl_root_fsolver_set(solver, &gsl_f, lo, hi);
        if (set_status != GSL_SUCCESS)
        {
            throw std::runtime_error("GSL rejected the Brent root bracket");
        }

        for (int step = 0; step < max_steps; ++step)
        {
            const int status = gsl_root_fsolver_iterate(solver);
            if (status != GSL_SUCCESS)
            {
                throw std::runtime_error("GSL Brent root iteration failed");
            }

            const double current_lo = gsl_root_fsolver_x_lower(solver);
            const double current_hi = gsl_root_fsolver_x_upper(solver);
            const double current_flo = f(current_lo);
            const double current_fhi = f(current_hi);
            if (!std::isfinite(current_flo) || !std::isfinite(current_fhi))
            {
                throw std::runtime_error("root function returned a non-finite value");
            }
            if (current_flo == 0.0)
                return {current_lo, current_lo};
            if (current_fhi == 0.0)
                return {current_hi, current_hi};
            if (root_converged(
                    current_lo, current_hi, current_flo, current_fhi,
                    x_tolerance, y_tolerance))
            {
                return {current_lo, current_hi};
            }
        }

        throw std::runtime_error(
            "GSL Brent root solver did not satisfy x/y tolerances");
#else
        (void)f;
        (void)lo;
        (void)hi;
        (void)flo;
        (void)fhi;
        (void)max_steps;
        (void)x_tolerance;
        (void)y_tolerance;
        throw std::runtime_error(
            "root_finder='brent' requires GSL at build time");
#endif
    }

    // Boost.Math TOMS 748 root finding.  The callback cache prevents the
    // tolerance predicate from re-evaluating the expensive wealth function.
    template <typename Function>
    std::pair<double, double> boost_root(
        Function &&f,
        double lo,
        double hi,
        double flo,
        double fhi,
        int max_steps,
        double x_tolerance,
        double y_tolerance,
        RootTraceRecorder* trace = nullptr)
    {
#if defined(ADAPTIVE_CS_HAS_BOOST)
        if (flo == 0.0)
            return {lo, lo};
        if (fhi == 0.0)
            return {hi, hi};

        std::map<double, double> cache{{lo, flo}, {hi, fhi}};
        auto evaluate = [&](double x)
        {
            const auto found = cache.find(x);
            if (found != cache.end())
                return found->second;
            const double value = f(x);
            if (!std::isfinite(value))
            {
                throw std::runtime_error("root function returned a non-finite value");
            }
            cache.emplace(x, value);
            if (trace != nullptr)
            {
                trace->observe(x, value, cache.size());
            }
            return value;
        };
        auto tolerance = [&](double a, double b)
        {
            if (std::abs(b - a) > x_tolerance)
                return false;
            const double fa = evaluate(a);
            const double fb = evaluate(b);
            return std::min(std::abs(fa), std::abs(fb)) <= y_tolerance;
        };

        std::uintmax_t max_iter = static_cast<std::uintmax_t>(max_steps);
        const auto bracket = boost::math::tools::toms748_solve(
            evaluate, lo, hi, flo, fhi, tolerance, max_iter);
        const double bracket_lo = bracket.first;
        const double bracket_hi = bracket.second;
        const double bracket_flo = evaluate(bracket_lo);
        const double bracket_fhi = evaluate(bracket_hi);
        if (bracket_flo == 0.0)
            return {bracket_lo, bracket_lo};
        if (bracket_fhi == 0.0)
            return {bracket_hi, bracket_hi};
        if (!root_converged(
                bracket_lo, bracket_hi, bracket_flo, bracket_fhi,
                x_tolerance, y_tolerance))
        {
            throw std::runtime_error(
                "Boost TOMS748 root solver did not satisfy x/y tolerances");
        }
        return {bracket_lo, bracket_hi};
#else
        (void)f;
        (void)lo;
        (void)hi;
        (void)flo;
        (void)fhi;
        (void)max_steps;
        (void)x_tolerance;
        (void)y_tolerance;
        throw std::runtime_error(
            "root_finder='boost' requires Boost.Math at build time");
#endif
    }

    template <typename Function>
    std::pair<double, double> solve_root(
        Function &&f,
        double lo,
        double hi,
        double flo,
        double fhi,
        int max_steps,
        double x_tolerance,
        double y_tolerance,
        const std::string &root_finder)
    {
        validate_root_bracket(lo, hi, flo, fhi);
        if (root_finder == "bisection")
        {
            return bisection_root(
                std::forward<Function>(f), lo, hi, flo, fhi,
                max_steps, x_tolerance, y_tolerance);
        }
        if (root_finder == "boost")
        {
            return boost_root(
                std::forward<Function>(f), lo, hi, flo, fhi,
                max_steps, x_tolerance, y_tolerance);
        }
        return brent_root(
            std::forward<Function>(f), lo, hi, flo, fhi,
            max_steps, x_tolerance, y_tolerance);
    }

} // namespace

// ---------------------------------------------------------------------------
// Single-tracker confidence-sequence implementation.
// ---------------------------------------------------------------------------

namespace adaptive_cs // contains the actual library classes and their method implementations (e.g., ConfidenceSequence, BatchedTracker)
{

    ConfidenceSequence::ConfidenceSequence(ConfidenceSequenceSettings settings)
        : settings_(std::move(settings))
    {
        validate_settings();
    }

    void ConfidenceSequence::validate_settings() const
    {
        if (!(settings_.alpha > 0.0 && settings_.alpha < 1.0))
        {
            throw std::invalid_argument("alpha must lie in (0, 1)");
        }
        if (!(settings_.c > 0.0 && settings_.c < 1.0))
        {
            throw std::invalid_argument("c must lie in (0, 1)");
        }
        if (!(settings_.theta > 0.0 && settings_.theta < 1.0))
        {
            throw std::invalid_argument("theta must lie in (0, 1)");
        }
        if (!(settings_.x_tolerance > 0.0 && settings_.y_tolerance > 0.0))
        {
            throw std::invalid_argument("root-finding tolerances must be positive");
        }
        if (settings_.max_steps < 1)
        {
            throw std::invalid_argument("max_steps must be positive");
        }
        if (settings_.endpoint_root_mode != "full" &&
            settings_.endpoint_root_mode != "componentwise")
        {
            throw std::invalid_argument(
                "endpoint_root_mode must be 'full' or 'componentwise'");
        }
        if (settings_.root_finder != "brent" &&
            settings_.root_finder != "boost" &&
            settings_.root_finder != "bisection")
        {
            throw std::invalid_argument(
                "root_finder must be 'brent', 'boost', or 'bisection'");
        }
    }

    void ConfidenceSequence::validate_value(const std::vector<double> &value) const
    {
        if (value.size() < 2)
        {
            throw std::invalid_argument(
                "value must be a one-dimensional probability vector");
        }
        if (n_classes_ != 0 && value.size() != n_classes_)
        {
            throw std::invalid_argument("the number of classes cannot change");
        }
        double total = 0.0;
        for (double entry : value)
        {
            if (!std::isfinite(entry) || entry < 0.0)
            {
                throw std::invalid_argument(
                    "value must be a finite probability vector summing to one");
            }
            total += entry;
        }
        if (!settings_.validate_probabilities)
        {
            return;
        }
        // Match numpy.isclose(sum, 1.0) used by the Python reference.
        if (std::abs(total - 1.0) > 1e-8 + 1e-5 * std::abs(total))
        {
            throw std::invalid_argument(
                "value must be a finite probability vector summing to one");
        }
    }

    std::pair<double, double> ConfidenceSequence::log_components(
        std::size_t cls,
        double candidate) const
    {
        const double plus_cap = candidate == 0.0
                                    ? std::numeric_limits<double>::infinity()
                                    : settings_.c / candidate;
        const double minus_cap = candidate == 1.0
                                     ? std::numeric_limits<double>::infinity()
                                     : settings_.c / (1.0 - candidate);
        double plus = log_initial_capital(settings_.theta, kPlus);
        double minus = log_initial_capital(settings_.theta, kMinus);
        for (std::size_t sample = 0; sample < n_samples_; ++sample)
        {
            const std::size_t offset = sample * n_classes_ + cls;
            const double centered = history_[offset] - candidate;
            const double plus_stake = std::min(bets_[offset], plus_cap);
            const double minus_stake = std::min(bets_[offset], minus_cap);
            plus += std::log1p(plus_stake * centered);
            minus += std::log1p(-minus_stake * centered);
        }
        return {plus, minus};
    }

    double ConfidenceSequence::log_component(
        std::size_t cls,
        double candidate,
        int component) const
    {
        // Endpoint inversion in componentwise mode only needs one directional
        // capital process. Do not call log_components() here: that routine
        // intentionally evaluates both directions for the full-wealth path.
        const bool plus = component == kPlus;
        const double denominator = plus ? candidate : 1.0 - candidate;
        const double cap = denominator == 0.0
                               ? std::numeric_limits<double>::infinity()
                               : settings_.c / denominator;
        double log_value = log_initial_capital(settings_.theta, component);
        for (std::size_t sample = 0; sample < n_samples_; ++sample)
        {
            const std::size_t offset = sample * n_classes_ + cls;
            const double stake = std::min(bets_[offset], cap);
            const double centered = history_[offset] - candidate;
            log_value += std::log1p((plus ? stake : -stake) * centered);
        }
        return log_value;
    }

    double ConfidenceSequence::component_wealth(
        std::size_t cls,
        double candidate,
        int component) const
    {
        return safe_exp(log_component(cls, candidate, component));
    }

    double ConfidenceSequence::wealth(std::size_t cls, double candidate) const
    {
        const auto logs = log_components(cls, candidate);
        return safe_exp(std::max(logs.first, logs.second));
    }

    ConfidenceSequence::MinimumBracket ConfidenceSequence::minimum(std::size_t cls) const
    {
        double left = 0.0;
        double right = 1.0;
        auto left_logs = log_components(cls, left);
        auto right_logs = log_components(cls, right);
        double left_plus = left_logs.first;
        double left_minus = left_logs.second;
        double right_plus = right_logs.first;
        double right_minus = right_logs.second;

        const bool boundary = left_minus > left_plus || right_minus < right_plus;
        const double left_wealth = safe_exp(std::max(left_plus, left_minus));
        const double right_wealth = safe_exp(std::max(right_plus, right_minus));
        if (boundary)
        {
            return {
                std::min(left_wealth, right_wealth),
                std::min(left_wealth, right_wealth),
                left_wealth <= right_wealth ? 0.0 : 1.0,
                left_wealth <= right_wealth ? 0.0 : 1.0,
                left_wealth <= right_wealth ? 0.0 : 1.0,
            };
        }

        // g(m) = log(k-) - log(k+) is non-decreasing.  Its zero is the
        // crossing at which max(k+, k-) is minimized.
        const auto crossing = [&](double candidate)
        {
            const auto logs = log_components(cls, candidate);
            return logs.second - logs.first;
        };
        const double left_crossing = left_minus - left_plus;
        const double right_crossing = right_minus - right_plus;
        const auto bracket = solve_root(
            crossing,
            left,
            right,
            left_crossing,
            right_crossing,
            settings_.max_steps,
            settings_.x_tolerance,
            settings_.y_tolerance,
            settings_.root_finder);
        left = bracket.first;
        right = bracket.second;

        left_logs = log_components(cls, left);
        right_logs = log_components(cls, right);
        left_plus = left_logs.first;
        left_minus = left_logs.second;
        right_plus = right_logs.first;
        right_minus = right_logs.second;

        return {
            std::max(safe_exp(left_minus), safe_exp(right_plus)), // lower wealth
            std::min(safe_exp(right_minus), safe_exp(left_plus)), // upper wealth
            0.5 * (left + right),                                 // point
            left,                                                 // left endpoint
            right,                                                // right endpoint
        };
    }

    double ConfidenceSequence::endpoint(
        std::size_t cls,
        double level,
        double center,
        bool lower) const
    {
        // endpoint() finds one side of the confidence interval for a particular class.
        // The 'level' parameter specifies the wealth threshold for the confidence interval.
        // The 'center' parameter is a reference point for the search.
        // The 'lower' parameter indicates whether we are computing the lower or upper endpoint.

        // The root is found from log(level) - log(wealth(cls, candidate)) = 0;
        // lower/upper determines the component and search direction.

        // Choose the direction
        const bool componentwise = settings_.endpoint_root_mode == "componentwise";
        const int component = lower ? kPlus : kMinus;

        // used for checks lateron whether level is outside the feasible range
        const double outside = lower ? 0.0 : 1.0;

        // Define the evaluation function for the candidate point.
        const auto evaluate = [&](double candidate)
        {
            if (componentwise)
            {
                return component_wealth(cls, candidate, component);
            }
            return wealth(cls, candidate);
        };

        // If the boundary is already feasible, the interval reaches that
        // boundary.
        if (evaluate(outside) <= level)
        {
            return outside;
        }

        // The minimum supplies a feasible center.  Keep this guard for
        // numerical inconsistencies before passing the bracket to solve_root.
        if (evaluate(center) > level)
        {
            // The caller normally catches this using the minimum.  Returning the
            // outside endpoint makes the failure conservative if rounding leaves
            // a tiny numerical mismatch.
            return outside;
        }

        // Search in an increasing coordinate z.  For the lower endpoint,
        // candidate = z; for the upper endpoint, candidate = 1 - z.
        // Stop at the known-feasible center so that the search stays on the
        // relevant side of the wealth minimum.  Thus the candidate interval is
        // [0, center] for a lower endpoint and [center, 1] for an upper one.
        const double search_hi = lower ? center : 1.0 - center;
        const auto f = [&](double z)
        {
            const double candidate = lower ? z : 1.0 - z;
            if (componentwise)
            {
                return std::log(level) -
                       log_component(cls, candidate, component);
            }
            const auto logs = log_components(cls, candidate);
            return std::log(level) - std::max(logs.first, logs.second);
        };

        const double f_lo = f(0.0);
        const double f_hi = f(search_hi);
        const auto bracket = solve_root(
            f, 0.0, search_hi, f_lo, f_hi,
            settings_.max_steps, settings_.x_tolerance,
            settings_.y_tolerance, settings_.root_finder);

        // solve_root returns the conservative outside side of the bracket.
        // Map the transformed upper-endpoint coordinate back to the candidate.
        return lower ? bracket.first : 1.0 - bracket.first;
    }

    std::vector<double> ConfidenceSequence::confidence_levels(
        const std::vector<MinimumBracket> &minima) const
    {
        double average_minimum = 0.0;
        for (const MinimumBracket &minimum : minima)
        {
            average_minimum += minimum.lower_wealth;
        }
        average_minimum /= static_cast<double>(n_classes_);

        std::vector<double> levels(n_classes_);
        for (std::size_t cls = 0; cls < n_classes_; ++cls)
        {
            levels[cls] = static_cast<double>(n_classes_) * (1.0 / settings_.alpha - average_minimum +
                                                             minima[cls].lower_wealth / static_cast<double>(n_classes_) +
                                                             settings_.padding);
        }
        return levels;
    }

    void ConfidenceSequence::intersect_time(Result &result) const
    {
        // A confidence sequence covers the target at every time on one joint
        // event. Intersecting every box observed so far therefore preserves
        // coverage and makes the returned endpoints nested in time.
        if (current_result_.lower.size() != n_classes_ ||
            current_result_.upper.size() != n_classes_)
        {
            return;
        }
        for (std::size_t cls = 0; cls < n_classes_; ++cls)
        {
            result.lower[cls] = std::max(result.lower[cls], current_result_.lower[cls]);
            result.upper[cls] = std::min(result.upper[cls], current_result_.upper[cls]);
        }
    }

    void ConfidenceSequence::intersect_simplex(Result &result) const
    {
        if (!settings_.simplex_intersection || !settings_.validate_probabilities)
        {
            return;
        }

        const std::vector<double> old_lower = result.lower;
        const std::vector<double> old_upper = result.upper;
        const double lower_sum = std::accumulate(old_lower.begin(), old_lower.end(), 0.0);
        const double upper_sum = std::accumulate(old_upper.begin(), old_upper.end(), 0.0);
        for (std::size_t cls = 0; cls < n_classes_; ++cls)
        {
            result.lower[cls] = std::clamp(
                std::max(old_lower[cls], 1.0 - (upper_sum - old_upper[cls])), 0.0, 1.0);
            result.upper[cls] = std::clamp(
                std::min(old_upper[cls], 1.0 - (lower_sum - old_lower[cls])), 0.0, 1.0);
        }
    }

    Result ConfidenceSequence::bounds(bool parallel_classes) const
    {
        // Computes a confidence interval for every class using the observations stored so far. Conceptually:
        // for each class:
        //     find the candidate mean with the smallest wealth
        // use those minima to set confidence thresholds
        // for each class:
        //     find lower and upper candidates whose wealth reaches the threshold
        // enforce that all class means sum to one
        // return the intervals

        // parallel_classes: whether to compute in parallel

        // Find the minimum wealth for each class
        std::vector<MinimumBracket> minima(n_classes_); // length: n_classes_
        std::exception_ptr error;

#ifdef ADAPTIVE_CS_OPENMP
#pragma omp parallel for if (parallel_classes) schedule(static)
#endif
        for (long long cls = 0; cls < static_cast<long long>(n_classes_); ++cls)
        {
            try
            {
                minima[static_cast<std::size_t>(cls)] = minimum(static_cast<std::size_t>(cls));
            }
            catch (...)
            {
#ifdef ADAPTIVE_CS_OPENMP
#pragma omp critical(adaptive_cs_error)
#endif
                {
                    if (!error)
                        error = std::current_exception();
                }
            }
        }
        if (error)
            std::rethrow_exception(error);

        const std::vector<double> levels = confidence_levels(minima);

        Result result;
        result.lower.resize(n_classes_);
        result.upper.resize(n_classes_);
#ifdef ADAPTIVE_CS_OPENMP
#pragma omp parallel for if (parallel_classes) schedule(static)
#endif
        // Compute each class's interval
        for (long long cls = 0; cls < static_cast<long long>(n_classes_); ++cls)
        {
            try
            {
                const std::size_t index = static_cast<std::size_t>(cls);
                const MinimumBracket &min = minima[index];

                // First, it checks whether even the minimum possible wealth is too large:
                if (min.lower_wealth > levels[index])
                {
                    result.lower[index] = 1.0;
                    result.upper[index] = 0.0;
                    continue;
                }

                // handles possible numerical error from approximating the minimum
                const double center_wealth = wealth(index, min.point);
                if (center_wealth > levels[index])
                {
                    result.lower[index] = 1.0;
                    result.upper[index] = 0.0;
                    continue;
                }

                // Compute the lower and upper endpoints for the confidence interval
                result.lower[index] = endpoint(
                    index, levels[index], min.point, true);
                result.upper[index] = endpoint(
                    index, levels[index], min.point, false);
            }
            catch (...)
            {
#ifdef ADAPTIVE_CS_OPENMP
#pragma omp critical(adaptive_cs_error)
#endif
                {
                    if (!error)
                        error = std::current_exception();
                }
            }
        }
        if (error)
            std::rethrow_exception(error);
        intersect_simplex(result);
        return result;
    }

    Result ConfidenceSequence::current_result() const
    {
        return current_result_;
    }

    Result ConfidenceSequence::update( // consumes one probability-vector observation and updates the tracker's state.
        const std::vector<double> &value,
        bool compute_bounds,
        bool parallel_bounds)
    {
        // recall there is a Confidence Sequence for each input, so no need to think about batches

        // value: a porbability vector of length n_classes_
        // compute_bounds: whether to recompute the confidence bounds after this update or only update the internal state.
        // parallel_bounds: whether to compute bounds in parallel across classes.

        validate_value(value);
        if (n_classes_ == 0)
        {
            n_classes_ = value.size();
            sum_x_.assign(n_classes_, 0.0);
            sum_dev_.assign(n_classes_, 0.0);
            variance_.assign(n_classes_, 0.25);
        }

        const std::size_t t = n_samples_ + 1;
        const double numerator = 2.0 * std::log(2.0 / settings_.alpha);
        const double denominator_factor = static_cast<double>(t) *
                                          std::log1p(static_cast<double>(t));

        history_.insert(history_.end(), value.begin(), value.end());
        bets_.resize(bets_.size() + n_classes_);
        for (std::size_t cls = 0; cls < n_classes_; ++cls)
        {
            const double denominator = variance_[cls] * denominator_factor;
            bets_[bets_.size() - n_classes_ + cls] = std::sqrt(
                numerator / std::max(denominator, kMinimumVariance));
        }

        ++n_samples_;
        for (std::size_t cls = 0; cls < n_classes_; ++cls)
        {
            const double regularized_mean =
                (0.5 + sum_x_[cls] + value[cls]) /
                (static_cast<double>(t) + 1.0);
            const double deviation = value[cls] - regularized_mean;
            sum_dev_[cls] += deviation * deviation;
            sum_x_[cls] += value[cls];
            variance_[cls] = (0.25 + sum_dev_[cls]) /
                             (static_cast<double>(t) + 1.0);
        }
        if (compute_bounds || n_samples_ == 1)
        {
            // Keep the previous box available while intersecting the newly
            // inverted one; assigning first would discard the history.
            Result result = bounds(parallel_bounds);
            intersect_time(result);
            current_result_ = std::move(result);
        }
        return current_result_;
    }

    double ConfidenceSequence::envelope_component_wealth(
        std::size_t cls, double candidate, int component, std::size_t horizon) const
    {
        const double cap = candidate == (component == kPlus ? 0.0 : 1.0)
                               ? std::numeric_limits<double>::infinity()
                               : settings_.c / (component == kPlus ? candidate : 1.0 - candidate);
        double log_value = log_initial_capital(settings_.theta, component);
        for (std::size_t sample = 0; sample < n_samples_; ++sample)
        {
            const std::size_t offset = sample * n_classes_ + cls;
            const double stake = std::min(bets_[offset], cap);
            const double centered = history_[offset] - candidate;
            log_value += std::log1p((component == kPlus ? stake : -stake) * centered);
        }
        const double numerator = 2.0 * std::log(2.0 / settings_.alpha);
        const double variance_numerator = std::max(0.25 + sum_dev_[cls], kMinimumVariance);
        for (std::size_t time = n_samples_ + 1; time <= horizon; ++time)
        {
            const double stake = std::min(
                std::sqrt(numerator / (variance_numerator * std::log1p(static_cast<double>(time)))),
                cap);
            const double favorable = component == kPlus ? 1.0 - candidate : candidate;
            log_value += std::log1p(stake * favorable);
        }
        return safe_exp(log_value);
    }

    double ConfidenceSequence::envelope_wealth(
        std::size_t cls, double candidate, std::size_t horizon) const
    {
        return std::max(
            envelope_component_wealth(cls, candidate, kPlus, horizon),
            envelope_component_wealth(cls, candidate, kMinus, horizon));
    }

    ConfidenceSequence::MinimumBracket ConfidenceSequence::envelope_minimum(
        std::size_t cls, std::size_t horizon) const
    {
        double left = 0.0, right = 1.0;
        auto crossing = [&](double candidate)
        {
            const double plus = std::log(envelope_component_wealth(cls, candidate, kPlus, horizon));
            const double minus = std::log(envelope_component_wealth(cls, candidate, kMinus, horizon));
            return minus - plus;
        };
        const double left_crossing = crossing(left);
        const double right_crossing = crossing(right);
        const double left_wealth = envelope_wealth(cls, left, horizon);
        const double right_wealth = envelope_wealth(cls, right, horizon);
        if (left_crossing > 0.0 || right_crossing < 0.0)
        {
            const double endpoint = left_wealth <= right_wealth ? 0.0 : 1.0;
            const double value = std::min(left_wealth, right_wealth);
            return {value, value, endpoint, endpoint, endpoint};
        }
        const auto bracket = solve_root(
            crossing, left, right, left_crossing, right_crossing,
            settings_.max_steps, settings_.x_tolerance,
            settings_.y_tolerance, settings_.root_finder);
        left = bracket.first;
        right = bracket.second;
        const double left_plus = envelope_component_wealth(cls, left, kPlus, horizon);
        const double left_minus = envelope_component_wealth(cls, left, kMinus, horizon);
        const double right_plus = envelope_component_wealth(cls, right, kPlus, horizon);
        const double right_minus = envelope_component_wealth(cls, right, kMinus, horizon);
        return {
            std::max(left_minus, right_plus),
            std::min(right_minus, left_plus),
            0.5 * (left + right),
            left,
            right,
        };
    }

    double ConfidenceSequence::envelope_endpoint(
        std::size_t cls, double level, double center, bool lower, std::size_t horizon) const
    {
        const bool componentwise = settings_.endpoint_root_mode == "componentwise";
        const int component = lower ? kPlus : kMinus;
        const double outside = lower ? 0.0 : 1.0;
        const double inside = lower ? 1.0 : 0.0;
        const auto evaluate = [&](double candidate)
        {
            return componentwise
                       ? envelope_component_wealth(cls, candidate, component, horizon)
                       : envelope_wealth(cls, candidate, horizon);
        };
        if (evaluate(outside) <= level)
            return outside;
        if (evaluate(inside) > level)
            return outside;
        const auto f = [&](double z)
        {
            const double candidate = lower ? z : 1.0 - z;
            return std::log(level) - std::log(
                                         componentwise
                                             ? envelope_component_wealth(cls, candidate, component, horizon)
                                             : envelope_wealth(cls, candidate, horizon));
        };
        const auto bracket = solve_root(
            f, 0.0, lower ? center : 1.0 - center,
            f(0.0), f(lower ? center : 1.0 - center),
            settings_.max_steps, settings_.x_tolerance,
            settings_.y_tolerance, settings_.root_finder);
        return lower ? bracket.first : 1.0 - bracket.first;
    }

    std::pair<std::vector<double>, std::vector<double>> ConfidenceSequence::envelope_bounds(
        std::size_t horizon) const
    {
        std::vector<MinimumBracket> minima(n_classes_);
#ifdef ADAPTIVE_CS_OPENMP
        const bool parallel_classes =
            !omp_in_parallel() &&
            n_classes_ >= static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#pragma omp parallel for if (parallel_classes) schedule(static)
        for (long long raw_cls = 0; raw_cls < static_cast<long long>(n_classes_); ++raw_cls)
        {
            const std::size_t cls = static_cast<std::size_t>(raw_cls);
            minima[cls] = envelope_minimum(cls, horizon);
        }
#else
        for (std::size_t cls = 0; cls < n_classes_; ++cls)
        {
            minima[cls] = envelope_minimum(cls, horizon);
        }
#endif
        const std::vector<double> levels = confidence_levels(minima);
        std::vector<double> lower(n_classes_), upper(n_classes_);
#ifdef ADAPTIVE_CS_OPENMP
#pragma omp parallel for if (parallel_classes) schedule(static)
        for (long long raw_cls = 0; raw_cls < static_cast<long long>(n_classes_); ++raw_cls)
        {
            const std::size_t cls = static_cast<std::size_t>(raw_cls);
            const double level = levels[cls];
            if (minima[cls].lower_wealth > level ||
                envelope_wealth(cls, minima[cls].point, horizon) > level)
            {
                lower[cls] = 1.0;
                upper[cls] = 0.0;
            }
            else
            {
                lower[cls] = envelope_endpoint(cls, level, minima[cls].point, true, horizon);
                upper[cls] = envelope_endpoint(cls, level, minima[cls].point, false, horizon);
            }
        }
#else
        for (std::size_t cls = 0; cls < n_classes_; ++cls)
        {
            const double level = levels[cls];
            if (minima[cls].lower_wealth > level ||
                envelope_wealth(cls, minima[cls].point, horizon) > level)
            {
                lower[cls] = 1.0;
                upper[cls] = 0.0;
            }
            else
            {
                lower[cls] = envelope_endpoint(cls, level, minima[cls].point, true, horizon);
                upper[cls] = envelope_endpoint(cls, level, minima[cls].point, false, horizon);
            }
        }
#endif
        Result result{std::move(lower), std::move(upper)};
        intersect_simplex(result);
        // Futility must use every bound already retained by the stopping rule.
        intersect_time(result);
        return {std::move(result.lower), std::move(result.upper)};
    }

    bool ConfidenceSequence::futile(
        int criterion, double tolerance, double criterion_level, std::size_t max_samples) const
    {
        if (criterion != ACS_FUTILITY_CLASS &&
            criterion != ACS_FUTILITY_DISTRIBUTION &&
            criterion != ACS_FUTILITY_THRESHOLD)
        {
            throw std::invalid_argument("unknown futility criterion");
        }
        if (n_samples_ == 0 || n_samples_ >= max_samples || current_result_.lower.empty())
        {
            return false;
        }
        if (criterion == ACS_FUTILITY_THRESHOLD)
        { // the cap envelope is sufficient.
            const auto envelope = envelope_bounds(max_samples);
            for (std::size_t cls = 0; cls < n_classes_; ++cls)
            {
                if (envelope.first[cls] >= criterion_level - tolerance)
                    return false;
            }
            for (double value : envelope.second)
            {
                if (value >= criterion_level + tolerance)
                    return false;
            }
            return true;
        }
        if (criterion == ACS_FUTILITY_CLASS)
        { // class selection.
            const auto envelope = envelope_bounds(max_samples);
            for (std::size_t cls = 0; cls < n_classes_; ++cls)
            {
                double competitor = -std::numeric_limits<double>::infinity();
                for (std::size_t other = 0; other < n_classes_; ++other)
                {
                    if (other != cls)
                        competitor = std::max(competitor, envelope.second[other]);
                }
                if (envelope.first[cls] + tolerance > competitor)
                    return false;
            }
            return true;
        }
        // Distribution accuracy: evaluate the optimistic box at the cap,
        // while accounting for every empirical mean reachable at that cap.
        const double n = static_cast<double>(n_samples_);
        std::vector<double> mean(n_classes_);
        for (std::size_t cls = 0; cls < n_classes_; ++cls)
            mean[cls] = sum_x_[cls] / n;

        const std::size_t horizon = max_samples;
        const auto envelope = envelope_bounds(horizon);
        const double remaining = static_cast<double>(horizon - n_samples_);
        bool feasible = true;
        double lower_sum = 0.0, upper_sum = 0.0;
        for (std::size_t cls = 0; cls < n_classes_; ++cls)
        {
            const double reachable_lower = n * mean[cls] / static_cast<double>(horizon);
            const double reachable_upper =
                (n * mean[cls] + remaining) / static_cast<double>(horizon);
            const double center_lower = std::max(
                reachable_lower, envelope.second[cls] - criterion_level);
            const double center_upper = std::min(
                reachable_upper, envelope.first[cls] + criterion_level);
            feasible &= center_lower <= center_upper + tolerance;
            lower_sum += center_lower;
            upper_sum += center_upper;
        }
        feasible &= lower_sum <= 1.0 + tolerance && upper_sum >= 1.0 - tolerance;
        return !feasible;
    }

    // Batch processing reuses the single-tracker implementation above.  It only
    // handles contiguous memory layouts and skips rows that have already stopped.
    BatchedTracker::BatchedTracker(
        std::size_t batch_size,
        ConfidenceSequenceSettings settings)
        : batch_size_(batch_size)
    {
        if (batch_size_ == 0)
        {
            throw std::invalid_argument("batch_size must be positive");
        }
        trackers_.reserve(batch_size_);
        for (std::size_t batch = 0; batch < batch_size_; ++batch)
        {
            trackers_.emplace_back(settings);
        }
    }

    void BatchedTracker::update_active( // updates only selected trackers in a batch (a tracker is a single ConfidenceSequence instance)
        const double *values,
        std::size_t n_draws,
        std::size_t n_classes,
        const std::size_t *active_indices,
        std::size_t n_active,
        bool compute_bounds,
        double *lower,
        double *upper)
    {
        // assume
        // values.shape = [n_draws, n_active, n_classes]
        // active_indices = [2, 5]

        // validate pointers
        if (values == nullptr || active_indices == nullptr || lower == nullptr || upper == nullptr)
        {
            throw std::invalid_argument("active update received a null pointer");
        }

        // validate dimensions
        if (n_draws == 0 || n_active == 0)
        {
            throw std::invalid_argument("active updates require positive dimensions");
        }

        // check if the number of classes has been set before and is consistent
        if (n_classes < 2)
            throw std::invalid_argument("n_classes must be at least two");
        if (n_classes_ != 0 && n_classes != n_classes_)
        {
            throw std::invalid_argument("the number of classes cannot change");
        }

        // Validate active indices:
        std::vector<bool> seen(batch_size_, false); // Create vector named `seen` containing batch_size_ elements, all initialized to false
        for (std::size_t i = 0; i < n_active; ++i)  // iterate over active indices and detect possible duplicates or out-of-range values
        {
            if (active_indices[i] >= batch_size_ || seen[active_indices[i]])
            {
                throw std::invalid_argument("active indices must be unique and in range");
            }
            seen[active_indices[i]] = true;
        }

        // Process every draw and active tracker
        // Draws must be processed in order because each tracker is stateful.
        // Active trackers are independent, so they can be updated in parallel.
        std::exception_ptr error;
        for (std::size_t draw = 0; draw < n_draws; ++draw)
        {
            // Only the final draw recomputes bounds. For large class counts,
            // choose between parallelizing active trackers and parallelizing
            // classes within each tracker. Never use nested OpenMP regions.
            const bool keep_result = compute_bounds && draw + 1 == n_draws;
            const ParallelMode parallel_mode =
                choose_parallel_mode(n_active, n_classes, keep_result);
            const bool parallelize_batch = parallel_mode == ParallelMode::Batch;
            const bool parallelize_classes = parallel_mode == ParallelMode::Classes;
#ifdef ADAPTIVE_CS_OPENMP
#pragma omp parallel for if (parallelize_batch) schedule(static)
#endif
            for (long long compact = 0;
                 compact < static_cast<long long>(n_active);
                 ++compact)
            {
                try
                {
                    const std::size_t active_position = static_cast<std::size_t>(compact);
                    const std::size_t batch = active_indices[active_position];
                    const double *begin = values +
                                          (draw * n_active + active_position) * n_classes;
                    std::vector<double> input(begin, begin + n_classes);
                    trackers_[batch].update(
                        input, keep_result, parallelize_classes && !parallelize_batch);
                }
                catch (...)
                {
#ifdef ADAPTIVE_CS_OPENMP
#pragma omp critical(adaptive_cs_error)
#endif
                    {
                        if (!error)
                            error = std::current_exception();
                    }
                }
            }
        }

        if (error)
            std::rethrow_exception(error);

        // After processing all draws, copy the current results from each tracker to the output arrays.
        n_classes_ = n_classes;
        for (std::size_t batch = 0; batch < batch_size_; ++batch)
        {
            const Result result = trackers_[batch].current_result();
            if (result.lower.empty())
            {
                throw std::logic_error("inactive tracker has no initialized result");
            }
            std::copy(result.lower.begin(), result.lower.end(), lower + batch * n_classes);
            std::copy(result.upper.begin(), result.upper.end(), upper + batch * n_classes);
        }
        n_samples_ += n_draws;
    }

    void BatchedTracker::futile_active(
        const std::size_t *active_indices,
        std::size_t n_active,
        int criterion,
        double tolerance,
        double criterion_level,
        std::size_t max_samples,
        unsigned char *futile) const
    {
        if (active_indices == nullptr || futile == nullptr)
        {
            throw std::invalid_argument("futility check received a null pointer");
        }
        for (std::size_t i = 0; i < n_active; ++i)
        {
            if (active_indices[i] >= batch_size_)
            {
                throw std::invalid_argument("active index is out of range");
            }
        }
#ifdef ADAPTIVE_CS_OPENMP
#pragma omp parallel for if (n_active > 1) schedule(static)
        for (long long raw_i = 0; raw_i < static_cast<long long>(n_active); ++raw_i)
        {
            const std::size_t i = static_cast<std::size_t>(raw_i);
            futile[i] = trackers_[active_indices[i]].futile(
                            criterion, tolerance, criterion_level, max_samples)
                            ? 1
                            : 0;
        }
#else
        for (std::size_t i = 0; i < n_active; ++i)
        {
            futile[i] = trackers_[active_indices[i]].futile(
                            criterion, tolerance, criterion_level, max_samples)
                            ? 1
                            : 0;
        }
#endif
    }

} // namespace adaptive_cs

// ---------------------------------------------------------------------------
// C ABI adapter.  Exceptions never cross this boundary; Python receives a
// status code and can retrieve the thread-local error message below.
// ---------------------------------------------------------------------------

namespace // contains private bridge helpers such as convert_settings and internal error state.
{
    thread_local std::string g_last_error;

    adaptive_cs::ConfidenceSequenceSettings convert_settings(acs_settings settings)
    {
        adaptive_cs::ConfidenceSequenceSettings converted;
        converted.alpha = settings.alpha;
        converted.c = settings.c;
        converted.theta = settings.theta;
        converted.x_tolerance = settings.x_tolerance;
        converted.y_tolerance = settings.y_tolerance;
        converted.max_steps = settings.max_steps;
        converted.padding = settings.padding;
        converted.endpoint_root_mode = settings.endpoint_root_mode == 0
                                           ? "full"
                                           : "componentwise";
        converted.root_finder = settings.root_finder == 0
                                    ? "brent"
                                    : (settings.root_finder == 1 ? "bisection" : "boost");
        converted.simplex_intersection = settings.simplex_intersection != 0;
        converted.validate_probabilities = settings.validate_probabilities != 0;
        return converted;
    }
} // namespace

extern "C" // these functions form the small C-compatible API used by Python's ctypes bridge.
{

    void *acs_batch_create(acs_settings settings, std::size_t batch_size)
    {
        try
        {
            return new adaptive_cs::BatchedTracker(
                batch_size, convert_settings(settings));
        }
        catch (const std::exception &error)
        {
            g_last_error = error.what();
            return nullptr;
        }
        catch (...)
        {
            g_last_error = "unknown C++ exception";
            return nullptr;
        }
    }

    void acs_batch_destroy(void *handle)
    {
        delete static_cast<adaptive_cs::BatchedTracker *>(handle);
    }

    int acs_batch_update_active(
        void *handle,
        const double *values,
        std::size_t n_draws,
        std::size_t batch_size,
        std::size_t n_classes,
        const std::size_t *active_indices,
        std::size_t n_active,
        int compute_bounds,
        double *lower,
        double *upper)
    {
        try
        {
            if (handle == nullptr)
            {
                throw std::invalid_argument("C++ batched tracker is null");
            }

            // cast the handle to a BatchedTracker pointer
            auto *tracker = static_cast<adaptive_cs::BatchedTracker *>(handle);
            if (tracker->batch_size() != batch_size)
            {
                throw std::invalid_argument("the batch size cannot change");
            }

            // update only the active indices of the tracker
            tracker->update_active(
                values, n_draws, n_classes, active_indices, n_active,
                compute_bounds != 0, lower, upper);

            g_last_error.clear();
            return 0;
        }
        catch (const std::exception &error)
        {
            g_last_error = error.what();
            return 1;
        }
        catch (...)
        {
            g_last_error = "unknown C++ exception";
            return 1;
        }
    }

    int acs_batch_futile(
        void *handle,
        const std::size_t *active_indices,
        std::size_t n_active,
        int criterion,
        double tolerance,
        double criterion_level,
        std::size_t max_samples,
        unsigned char *futile)
    {
        try
        {
            if (handle == nullptr)
            {
                throw std::invalid_argument("C++ batched tracker is null");
            }
            static_cast<adaptive_cs::BatchedTracker *>(handle)->futile_active(
                active_indices, n_active, criterion, tolerance, criterion_level,
                max_samples, futile);
            g_last_error.clear();
            return 0;
        }
        catch (const std::exception &error)
        {
            g_last_error = error.what();
            return 1;
        }
        catch (...)
        {
            g_last_error = "unknown C++ exception";
            return 1;
        }
    }

    int acs_trace_hedged_root(
        const double *observations,
        const double *bets,
        std::size_t n_samples,
        int call_type,
        double level,
        double center,
        double c,
        double theta,
        double x_tolerance,
        double y_tolerance,
        int max_steps,
        double *bracket_lower,
        double *bracket_upper,
        std::size_t *function_evaluations,
        std::size_t capacity,
        std::size_t *point_count,
        std::size_t *total_function_evaluations,
        double *root_lower,
        double *root_upper,
        int *status)
    {
        try
        {
            if (observations == nullptr || bets == nullptr ||
                bracket_lower == nullptr || bracket_upper == nullptr ||
                function_evaluations == nullptr || point_count == nullptr ||
                total_function_evaluations == nullptr || root_lower == nullptr ||
                root_upper == nullptr || status == nullptr)
            {
                throw std::invalid_argument("null pointer passed to root trace diagnostic");
            }
            if (n_samples == 0 || capacity == 0)
            {
                throw std::invalid_argument("root trace requires samples and output capacity");
            }
            if (call_type < 0 || call_type > 2)
            {
                throw std::invalid_argument("root trace call_type must be 0, 1, or 2");
            }
            if (!(c > 0.0 && c < 1.0) || !(theta > 0.0 && theta < 1.0) ||
                !(x_tolerance > 0.0) || !(y_tolerance > 0.0) || max_steps < 1)
            {
                throw std::invalid_argument("invalid root trace settings");
            }

            const auto log_component = [&](double candidate, int component)
            {
                const bool plus = component == kPlus;
                const double denominator = plus ? candidate : 1.0 - candidate;
                const double cap = denominator == 0.0
                                       ? std::numeric_limits<double>::infinity()
                                       : c / denominator;
                double value = plus ? std::log(theta) : std::log1p(-theta);
                for (std::size_t sample = 0; sample < n_samples; ++sample)
                {
                    const double stake = std::min(bets[sample], cap);
                    const double centered = observations[sample] - candidate;
                    value += std::log1p((plus ? stake : -stake) * centered);
                }
                return value;
            };

            double lo = 0.0;
            double hi = 1.0;
            bool reverse_coordinate = false;
            std::function<double(double)> function;
            if (call_type == 0)
            {
                function = [&](double candidate)
                {
                    return log_component(candidate, kMinus) -
                           log_component(candidate, kPlus);
                };
            }
            else
            {
                if (!(std::isfinite(level) && level > 0.0) ||
                    !(center >= 0.0 && center <= 1.0))
                {
                    throw std::invalid_argument("invalid endpoint level or center");
                }
                const int component = call_type == 1 ? kPlus : kMinus;
                const double outside = call_type == 1 ? 0.0 : 1.0;
                const double log_level = std::log(level);
                if (log_component(outside, component) <= log_level)
                {
                    *point_count = 0;
                    *total_function_evaluations = 1;
                    *root_lower = *root_upper = outside;
                    *status = 1;
                    g_last_error.clear();
                    return 0;
                }
                if (log_component(center, component) > log_level)
                {
                    *point_count = 0;
                    *total_function_evaluations = 2;
                    *root_lower = *root_upper = outside;
                    *status = 2;
                    g_last_error.clear();
                    return 0;
                }
                reverse_coordinate = call_type == 2;
                hi = call_type == 1 ? center : 1.0 - center;
                function = [&, component, log_level](double coordinate)
                {
                    const double candidate = call_type == 1
                                                 ? coordinate
                                                 : 1.0 - coordinate;
                    return log_level - log_component(candidate, component);
                };
            }

            const double flo = function(lo);
            const double fhi = function(hi);
            validate_root_bracket(lo, hi, flo, fhi);
            RootTraceRecorder recorder(lo, hi, flo, fhi, reverse_coordinate);
            const auto bracket = boost_root(
                function, lo, hi, flo, fhi, max_steps,
                x_tolerance, y_tolerance, &recorder);
            const auto &points = recorder.points();
            if (points.size() > capacity)
            {
                throw std::invalid_argument("root trace output capacity is too small");
            }
            for (std::size_t index = 0; index < points.size(); ++index)
            {
                bracket_lower[index] = points[index].lower;
                bracket_upper[index] = points[index].upper;
                function_evaluations[index] = points[index].function_evaluations;
            }
            *point_count = points.size();
            *total_function_evaluations = points.back().function_evaluations;
            if (reverse_coordinate)
            {
                *root_lower = 1.0 - bracket.second;
                *root_upper = 1.0 - bracket.first;
            }
            else
            {
                *root_lower = bracket.first;
                *root_upper = bracket.second;
            }
            *status = 0;
            g_last_error.clear();
            return 0;
        }
        catch (const std::exception &error)
        {
            g_last_error = error.what();
            return 1;
        }
        catch (...)
        {
            g_last_error = "unknown C++ exception";
            return 1;
        }
    }

    const char *acs_last_error()
    {
        return g_last_error.c_str();
    }

} // extern "C"
