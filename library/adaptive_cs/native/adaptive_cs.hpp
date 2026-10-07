#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace adaptive_cs {

// ---------------------------------------------------------------------------
// C++ confidence-sequence API
// ---------------------------------------------------------------------------

// Statistical and numerical settings for one confidence-sequence tracker.
struct ConfidenceSequenceSettings {
    double alpha = 0.05;  // CS error level; also controls the adaptive bet size.
    double c = 0.75;  // Cap coefficient for positive/negative stakes.
    double theta = 0.5;  // Initial capital in K+; K- receives 1 - theta.
    double x_tolerance = 1e-4;  // Candidate-mean/root-coordinate tolerance.
    double y_tolerance = 1e-6;  // Log-wealth/residual tolerance for root finding.
    int max_steps = 80;  // Maximum iterations per root solve.
    double padding = 1e-12;  // Slack added to simultaneous box levels.
    bool simplex_intersection = true;  // Tighten bounds using sum-to-one constraints.
    std::string endpoint_root_mode = "componentwise";  // "full" or "componentwise".
    std::string root_finder = "boost";  // "brent", "boost", or "bisection".
    bool validate_probabilities = true;  // False for augmented uncertainty observations.
};

// A confidence box for the current mean of each class.
struct Result {
    std::vector<double> lower;
    std::vector<double> upper;
};

// Stateful confidence-sequence engine for one observed vector at a time.
class ConfidenceSequence {
public:
    explicit ConfidenceSequence(ConfidenceSequenceSettings settings = {});

    // Consume one observation.  Bounds are recomputed only when requested (or
    // for the first observation), but the betting state is always updated.
    Result update(
        const std::vector<double>& value,
        bool compute_bounds = true,
        bool parallel_bounds = true);

    // Return the most recently computed box.
    Result current_result() const;

    // Conservative optimistic-envelope test used by early abstention.
    bool futile(
        int criterion,
        double tolerance,
        double criterion_level,
        std::size_t max_samples) const;

private:
    struct MinimumBracket {
        double lower_wealth;
        double upper_wealth;
        double point;
        double left;
        double right;
    };

    ConfidenceSequenceSettings settings_;
    std::size_t n_classes_ = 0;
    std::size_t n_samples_ = 0;
    std::vector<double> history_;  // Row-major: [sample][class]. length: n_samples * n_classes
    std::vector<double> bets_;     // Row-major: [sample][class]. length: n_samples * n_classes


    std::vector<double> sum_x_; // length: n_classes
    std::vector<double> sum_dev_; // length: n_classes
    std::vector<double> variance_; // length: n_classes

    Result current_result_;

    void validate_settings() const;
    void validate_value(const std::vector<double>& value) const;

    std::pair<double, double> log_components(
        std::size_t cls, double candidate) const;
    double log_component(std::size_t cls, double candidate, int component) const;
    double wealth(std::size_t cls, double candidate) const;
    double component_wealth(std::size_t cls, double candidate, int component) const;

    MinimumBracket minimum(std::size_t cls) const;
    double endpoint(std::size_t cls, double level, double center, bool lower) const;
    double envelope_wealth(std::size_t cls, double candidate, std::size_t horizon) const;
    double envelope_component_wealth(
        std::size_t cls, double candidate, int component, std::size_t horizon) const;
    MinimumBracket envelope_minimum(std::size_t cls, std::size_t horizon) const;
    double envelope_endpoint(
        std::size_t cls, double level, double center, bool lower, std::size_t horizon) const;
    std::pair<std::vector<double>, std::vector<double>> envelope_bounds(
        std::size_t horizon) const;
    Result bounds(bool parallel_classes = true) const;
    std::vector<double> confidence_levels(
        const std::vector<MinimumBracket>& minima) const;
    void intersect_time(Result& result) const;
    void intersect_simplex(Result& result) const;
};

// A fixed batch of independent confidence-sequence trackers. The batch API is
// an optimization layer: each row has the same statistical behavior as a
// standalone ConfidenceSequence.
class BatchedTracker {
public:
    BatchedTracker(
        std::size_t batch_size,
        ConfidenceSequenceSettings settings = {});

    void update_active(
        const double* values,
        std::size_t n_draws,
        std::size_t n_classes,
        const std::size_t* active_indices,
        std::size_t n_active,
        bool compute_bounds,
        double* lower,
        double* upper);

    void futile_active(
        const std::size_t* active_indices,
        std::size_t n_active,
        int criterion,
        double tolerance,
        double criterion_level,
        std::size_t max_samples,
        unsigned char* futile) const;

    std::size_t batch_size() const noexcept { return batch_size_; }
private:
    std::size_t batch_size_;
    std::size_t n_classes_ = 0;
    std::size_t n_samples_ = 0;
    std::vector<ConfidenceSequence> trackers_;
};

}  // namespace adaptive_cs

// ---------------------------------------------------------------------------
// C ABI consumed by the private Python bridge
// ---------------------------------------------------------------------------

// C ABI used by the private Python bridge.  Keeping this boundary tiny avoids
// a dependency on pybind11 or another third-party binding at build time.
#if defined(_WIN32) && defined(ADAPTIVE_CS_BUILD)
#define ACS_API __declspec(dllexport)
#else
#define ACS_API
#endif

extern "C" {

struct acs_settings {
    double alpha;
    double c;
    double theta;
    double x_tolerance;
    double y_tolerance;
    int max_steps;
    double padding;
    int endpoint_root_mode;  // 0 = full, 1 = componentwise
    int root_finder;  // 0 = brent, 1 = bisection, 2 = boost/TOMS748
    int simplex_intersection;
    int validate_probabilities;
};

// Values accepted by acs_batch_futile. Keep these values stable: they are
// mirrored by the private Python ctypes bridge.
enum acs_futility_criterion {
    ACS_FUTILITY_CLASS = 0,
    ACS_FUTILITY_DISTRIBUTION = 1,
    ACS_FUTILITY_THRESHOLD = 2
};

ACS_API void* acs_batch_create(acs_settings settings, std::size_t batch_size);
ACS_API void acs_batch_destroy(void* handle);

ACS_API int acs_batch_update_active(
    void* handle,
    const double* values,
    std::size_t n_draws,
    std::size_t batch_size,
    std::size_t n_classes,
    const std::size_t* active_indices,
    std::size_t n_active,
    int compute_bounds,
    double* lower,
    double* upper
);

ACS_API int acs_batch_futile(
    void* handle,
    const std::size_t* active_indices,
    std::size_t n_active,
    int criterion,
    double tolerance,
    double criterion_level,
    std::size_t max_samples,
    unsigned char* futile
);

// Diagnostic entry point used by the paper's breakpoint experiment.  It runs
// the same Boost.Math TOMS 748 implementation as the production tracker and
// records the enclosing bracket after every accepted proposal.  call_type is
// 0 for the K+/K- crossing, 1 for a lower-endpoint K+ inversion, and 2 for an
// upper-endpoint K- inversion.  Endpoint calls use level and center; these
// arguments are ignored for crossings.  Status is 0 for a solver call, 1 when
// an endpoint reaches the domain boundary without a root call, and 2 when the
// supplied center is numerically infeasible.
ACS_API int acs_trace_hedged_root(
    const double* observations,
    const double* bets,
    std::size_t n_samples,
    int call_type,
    double level,
    double center,
    double c,
    double theta,
    double x_tolerance,
    double y_tolerance,
    int max_steps,
    double* bracket_lower,
    double* bracket_upper,
    std::size_t* function_evaluations,
    std::size_t capacity,
    std::size_t* point_count,
    std::size_t* total_function_evaluations,
    double* root_lower,
    double* root_upper,
    int* status
);

// The returned pointer remains valid until the next C++ call on this thread.
ACS_API const char* acs_last_error();

}
