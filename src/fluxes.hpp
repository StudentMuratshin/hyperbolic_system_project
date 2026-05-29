#pragma once

#include <string>

#include "state.hpp"

namespace swe {

enum class FluxKind {
    Rusanov,
    Hll,
    Roe
};

std::string flux_name(FluxKind kind);
State numerical_flux(FluxKind kind, const State& left, const State& right);

} // namespace swe
