#pragma once

#include <algorithm>
#include <cmath>

namespace swe {

constexpr double kGravity = 9.81;
constexpr double kHeightFloor = 1.0e-10;

struct State {
    double h = 0.0;
    double hu = 0.0;
};

inline State make_state_from_h_u(double h, double u)
{
    const double safe_h = std::max(h, kHeightFloor);
    return {safe_h, safe_h * u};
}

inline State operator+(const State& a, const State& b)
{
    return {a.h + b.h, a.hu + b.hu};
}

inline State operator-(const State& a, const State& b)
{
    return {a.h - b.h, a.hu - b.hu};
}

inline State operator*(double s, const State& a)
{
    return {s * a.h, s * a.hu};
}

inline State operator*(const State& a, double s)
{
    return s * a;
}

inline State operator/(const State& a, double s)
{
    return {a.h / s, a.hu / s};
}

inline double safe_height(const State& u)
{
    return std::max(u.h, kHeightFloor);
}

inline double velocity(const State& u)
{
    return u.hu / safe_height(u);
}

inline double wave_celerity(const State& u)
{
    return std::sqrt(kGravity * safe_height(u));
}

inline double max_signal_speed(const State& u)
{
    return std::abs(velocity(u)) + wave_celerity(u);
}

inline State physical_flux(const State& u)
{
    const double h = safe_height(u);
    const double vel = u.hu / h;
    return {u.hu, u.hu * vel + 0.5 * kGravity * h * h};
}

inline bool is_finite(const State& u)
{
    return std::isfinite(u.h) && std::isfinite(u.hu);
}

} // namespace swe
