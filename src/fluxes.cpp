#include "fluxes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace swe {
namespace {

State sanitized(State u)
{
    if (u.h < kHeightFloor) {
        u.h = kHeightFloor;
        u.hu = 0.0;
    }
    return u;
}

State rusanov_flux(const State& left, const State& right)
{
    const State ul = sanitized(left);
    const State ur = sanitized(right);
    const State fl = physical_flux(ul);
    const State fr = physical_flux(ur);
    const double a = std::max(max_signal_speed(ul), max_signal_speed(ur));
    return 0.5 * (fl + fr) - 0.5 * a * (ur - ul);
}

State hll_flux(const State& left, const State& right)
{
    const State ul = sanitized(left);
    const State ur = sanitized(right);
    const State fl = physical_flux(ul);
    const State fr = physical_flux(ur);

    const double u_l = velocity(ul);
    const double u_r = velocity(ur);
    const double c_l = wave_celerity(ul);
    const double c_r = wave_celerity(ur);

    const double s_l = std::min(u_l - c_l, u_r - c_r);
    const double s_r = std::max(u_l + c_l, u_r + c_r);

    if (0.0 <= s_l) {
        return fl;
    }
    if (s_r <= 0.0) {
        return fr;
    }
    return (s_r * fl - s_l * fr + s_l * s_r * (ur - ul)) / (s_r - s_l);
}

double entropy_abs(double lambda, double delta)
{
    const double a = std::abs(lambda);
    if (delta <= 0.0 || a >= delta) {
        return a;
    }
    return 0.5 * (a * a / delta + delta);
}

State roe_flux(const State& left, const State& right)
{
    const State ul = sanitized(left);
    const State ur = sanitized(right);
    const State fl = physical_flux(ul);
    const State fr = physical_flux(ur);

    const double h_l = safe_height(ul);
    const double h_r = safe_height(ur);
    const double u_l = velocity(ul);
    const double u_r = velocity(ur);
    const double sqrt_h_l = std::sqrt(h_l);
    const double sqrt_h_r = std::sqrt(h_r);

    const double u_tilde =
        (sqrt_h_l * u_l + sqrt_h_r * u_r) / (sqrt_h_l + sqrt_h_r);
    const double c_tilde = std::sqrt(0.5 * kGravity * (h_l + h_r));

    const double lambda_1 = u_tilde - c_tilde;
    const double lambda_2 = u_tilde + c_tilde;
    const double delta_entropy = 0.1 * c_tilde;

    const State du = ur - ul;
    const double alpha_1 = ((u_tilde + c_tilde) * du.h - du.hu) / (2.0 * c_tilde);
    const double alpha_2 = (du.hu - (u_tilde - c_tilde) * du.h) / (2.0 * c_tilde);

    const State r_1{1.0, u_tilde - c_tilde};
    const State r_2{1.0, u_tilde + c_tilde};

    const State dissipation =
        entropy_abs(lambda_1, delta_entropy) * alpha_1 * r_1 +
        entropy_abs(lambda_2, delta_entropy) * alpha_2 * r_2;

    return 0.5 * (fl + fr) - 0.5 * dissipation;
}

} // namespace

std::string flux_name(FluxKind kind)
{
    switch (kind) {
    case FluxKind::Rusanov:
        return "rusanov";
    case FluxKind::Hll:
        return "hll";
    case FluxKind::Roe:
        return "roe";
    }
    return "unknown";
}

State numerical_flux(FluxKind kind, const State& left, const State& right)
{
    switch (kind) {
    case FluxKind::Rusanov:
        return rusanov_flux(left, right);
    case FluxKind::Hll:
        return hll_flux(left, right);
    case FluxKind::Roe:
        return roe_flux(left, right);
    }
    throw std::invalid_argument("Unknown numerical flux kind");
}

} // namespace swe
