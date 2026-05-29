#pragma once

#include <functional>
#include <string>
#include <vector>

#include "fluxes.hpp"
#include "state.hpp"

namespace swe {

enum class BoundaryKind {
    Outflow,
    Periodic
};

struct SolverConfig {
    int cells = 400;
    double x_min = -1.0;
    double x_max = 1.0;
    double t_final = 0.2;
    double cfl = 0.45;
    FluxKind flux = FluxKind::Hll;
    BoundaryKind boundary = BoundaryKind::Outflow;
    int max_mass_samples = 250;
};

struct MassSample {
    double t = 0.0;
    double mass = 0.0;
};

struct SolverResult {
    std::vector<double> x;
    std::vector<State> u;
    std::vector<MassSample> mass_history;
    double x_min = 0.0;
    double x_max = 0.0;
    double dx = 0.0;
    double final_time = 0.0;
    double runtime_ms = 0.0;
    double min_h = 0.0;
    int steps = 0;
    int positivity_fixes = 0;
    bool stable = true;
    std::string message;
};

using InitialCondition = std::function<State(double)>;

SolverResult solve(const SolverConfig& config, const InitialCondition& initial_condition);
double total_mass(const std::vector<State>& u, int first, int last, double dx);

} // namespace swe
