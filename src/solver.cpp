#include "solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace swe {
namespace {

void validate_config(const SolverConfig& config)
{
    if (config.cells < 4) {
        throw std::invalid_argument("SolverConfig::cells must be at least 4");
    }
    if (!(config.x_min < config.x_max)) {
        throw std::invalid_argument("SolverConfig requires x_min < x_max");
    }
    if (config.t_final < 0.0) {
        throw std::invalid_argument("SolverConfig::t_final must be non-negative");
    }
    if (config.cfl <= 0.0) {
        throw std::invalid_argument("SolverConfig::cfl must be positive");
    }
}

void apply_boundary(std::vector<State>& u, BoundaryKind boundary)
{
    const int n = static_cast<int>(u.size()) - 2;
    if (boundary == BoundaryKind::Periodic) {
        u[0] = u[n];
        u[n + 1] = u[1];
        return;
    }
    u[0] = u[1];
    u[n + 1] = u[n];
}

State repaired(State u, SolverResult& result)
{
    if (!is_finite(u)) {
        result.stable = false;
        result.message = "non-finite state encountered";
        return {kHeightFloor, 0.0};
    }
    if (u.h < kHeightFloor) {
        ++result.positivity_fixes;
        u.h = kHeightFloor;
        u.hu = 0.0;
    }
    return u;
}

} // namespace

double total_mass(const std::vector<State>& u, int first, int last, double dx)
{
    double mass = 0.0;
    for (int i = first; i <= last; ++i) {
        mass += u[i].h * dx;
    }
    return mass;
}

SolverResult solve(const SolverConfig& config, const InitialCondition& initial_condition)
{
    validate_config(config);

    const auto started = std::chrono::steady_clock::now();
    const int n = config.cells;
    const double dx = (config.x_max - config.x_min) / static_cast<double>(n);

    SolverResult result;
    result.x_min = config.x_min;
    result.x_max = config.x_max;
    result.dx = dx;
    result.x.resize(n);

    std::vector<State> u(n + 2);
    std::vector<State> next(n + 2);
    std::vector<State> face_flux(n + 1);

    for (int i = 1; i <= n; ++i) {
        const double x = config.x_min + (static_cast<double>(i) - 0.5) * dx;
        result.x[i - 1] = x;
        u[i] = repaired(initial_condition(x), result);
    }
    apply_boundary(u, config.boundary);

    double t = 0.0;
    const double initial_mass = total_mass(u, 1, n, dx);
    result.mass_history.push_back({0.0, initial_mass});
    double next_mass_output = config.t_final / std::max(1, config.max_mass_samples);

    constexpr int kMaxSteps = 10'000'000;
    while (t < config.t_final && result.stable) {
        apply_boundary(u, config.boundary);

        double max_speed = 0.0;
        for (int i = 1; i <= n; ++i) {
            max_speed = std::max(max_speed, max_signal_speed(u[i]));
        }
        if (!std::isfinite(max_speed)) {
            result.stable = false;
            result.message = "non-finite wave speed encountered";
            break;
        }

        double dt = config.t_final - t;
        if (max_speed > 1.0e-14) {
            dt = std::min(dt, config.cfl * dx / max_speed);
        }

        for (int i = 0; i <= n; ++i) {
            face_flux[i] = numerical_flux(config.flux, u[i], u[i + 1]);
        }

        const double dt_dx = dt / dx;
        next = u;
        for (int i = 1; i <= n; ++i) {
            next[i] = repaired(u[i] - dt_dx * (face_flux[i] - face_flux[i - 1]), result);
        }

        u.swap(next);
        t += dt;
        ++result.steps;

        if (t + 1.0e-14 >= next_mass_output || t >= config.t_final) {
            result.mass_history.push_back({t, total_mass(u, 1, n, dx)});
            next_mass_output += config.t_final / std::max(1, config.max_mass_samples);
        }

        if (result.steps >= kMaxSteps) {
            result.stable = false;
            result.message = "maximum number of time steps exceeded";
            break;
        }
    }

    result.final_time = t;
    result.u.assign(u.begin() + 1, u.begin() + n + 1);
    result.min_h = std::numeric_limits<double>::infinity();
    for (const State& state : result.u) {
        result.min_h = std::min(result.min_h, state.h);
    }

    if (result.message.empty()) {
        result.message = result.stable ? "ok" : "failed";
    }

    const auto finished = std::chrono::steady_clock::now();
    result.runtime_ms =
        std::chrono::duration<double, std::milli>(finished - started).count();
    return result;
}

} // namespace swe
