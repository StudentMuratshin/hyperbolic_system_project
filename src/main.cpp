#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace {

using swe::BoundaryKind;
using swe::FluxKind;
using swe::InitialCondition;
using swe::SolverConfig;
using swe::SolverResult;
using swe::State;

constexpr double kPi = 3.141592653589793238462643383279502884;

struct TestCase {
    std::string id;
    std::string title;
    double x_min = -1.0;
    double x_max = 1.0;
    double t_final = 0.2;
    BoundaryKind boundary = BoundaryKind::Outflow;
    InitialCondition initial;
};

struct ErrorMetrics {
    double h_l1 = 0.0;
    double u_l1 = 0.0;
};

struct NamedResult {
    FluxKind method = FluxKind::Hll;
    SolverResult result;
};

struct PlotRect {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
};

struct ConvergencePoint {
    int cells = 0;
    double h_l1 = 0.0;
};

struct ConvergenceSeries {
    FluxKind method = FluxKind::Hll;
    std::vector<ConvergencePoint> points;
};

std::ofstream open_output(const std::filesystem::path& path)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot open output file: " + path.string());
    }
    out << std::setprecision(16);
    return out;
}

std::string format_number(double value)
{
    std::ostringstream out;
    out << std::setprecision(4) << value;
    return out.str();
}

std::string color_for(FluxKind method)
{
    switch (method) {
    case FluxKind::Rusanov:
        return "#D55E00";
    case FluxKind::Hll:
        return "#0072B2";
    case FluxKind::Roe:
        return "#009E73";
    }
    return "#222222";
}

std::string dash_for(FluxKind method)
{
    switch (method) {
    case FluxKind::Hll:
        return " stroke-dasharray=\"7 4\"";
    case FluxKind::Roe:
        return " stroke-dasharray=\"10 4 2 4\"";
    case FluxKind::Rusanov:
        return "";
    }
    return "";
}

double map_x(double value, const PlotRect& rect, double min_value, double max_value)
{
    return rect.x + (value - min_value) / (max_value - min_value) * rect.width;
}

double map_y(double value, const PlotRect& rect, double min_value, double max_value)
{
    return rect.y + rect.height -
           (value - min_value) / (max_value - min_value) * rect.height;
}

std::pair<double, double> expanded_range(double min_value, double max_value)
{
    if (std::abs(max_value - min_value) < 1.0e-14) {
        const double delta = std::max(1.0, std::abs(min_value)) * 0.1;
        return {min_value - delta, max_value + delta};
    }
    const double padding = 0.08 * (max_value - min_value);
    return {min_value - padding, max_value + padding};
}

void write_svg_axes(
    std::ofstream& out,
    const PlotRect& rect,
    const std::string& y_label,
    double x_min,
    double x_max,
    double y_min,
    double y_max)
{
    out << "<rect x=\"" << rect.x << "\" y=\"" << rect.y << "\" width=\""
        << rect.width << "\" height=\"" << rect.height
        << "\" fill=\"white\" stroke=\"#333\" stroke-width=\"1\"/>\n";

    for (int i = 1; i < 4; ++i) {
        const double gy = rect.y + rect.height * static_cast<double>(i) / 4.0;
        out << "<line x1=\"" << rect.x << "\" y1=\"" << gy << "\" x2=\""
            << (rect.x + rect.width) << "\" y2=\"" << gy
            << "\" stroke=\"#dddddd\" stroke-width=\"1\"/>\n";
    }

    out << "<text x=\"" << rect.x << "\" y=\"" << (rect.y + rect.height + 24.0)
        << "\" font-size=\"12\">" << format_number(x_min) << "</text>\n";
    out << "<text x=\"" << (rect.x + rect.width - 35.0) << "\" y=\""
        << (rect.y + rect.height + 24.0) << "\" font-size=\"12\">"
        << format_number(x_max) << "</text>\n";
    out << "<text x=\"" << (rect.x - 58.0) << "\" y=\"" << (rect.y + 12.0)
        << "\" font-size=\"12\">" << format_number(y_max) << "</text>\n";
    out << "<text x=\"" << (rect.x - 58.0) << "\" y=\""
        << (rect.y + rect.height) << "\" font-size=\"12\">"
        << format_number(y_min) << "</text>\n";
    out << "<text x=\"" << (rect.x - 48.0) << "\" y=\""
        << (rect.y + rect.height / 2.0) << "\" font-size=\"15\" font-weight=\"600\">"
        << y_label << "</text>\n";
}

void write_profile_panel(
    std::ofstream& out,
    const PlotRect& rect,
    const std::vector<NamedResult>& results,
    const std::string& y_label,
    bool use_velocity)
{
    const double x_min = results.front().result.x_min;
    const double x_max = results.front().result.x_max;
    double y_min = std::numeric_limits<double>::infinity();
    double y_max = -std::numeric_limits<double>::infinity();

    for (const NamedResult& named : results) {
        for (const State& state : named.result.u) {
            const double value = use_velocity ? swe::velocity(state) : state.h;
            y_min = std::min(y_min, value);
            y_max = std::max(y_max, value);
        }
    }
    const auto [plot_y_min, plot_y_max] = expanded_range(y_min, y_max);
    write_svg_axes(out, rect, y_label, x_min, x_max, plot_y_min, plot_y_max);

    for (const NamedResult& named : results) {
        out << "<polyline fill=\"none\" stroke=\"" << color_for(named.method)
            << "\" stroke-width=\"2\"" << dash_for(named.method) << " points=\"";
        for (std::size_t i = 0; i < named.result.u.size(); ++i) {
            const double value =
                use_velocity ? swe::velocity(named.result.u[i]) : named.result.u[i].h;
            out << map_x(named.result.x[i], rect, x_min, x_max) << ','
                << map_y(value, rect, plot_y_min, plot_y_max) << ' ';
        }
        out << "\"/>\n";
    }
}

void write_profile_svg(
    const std::filesystem::path& path,
    const TestCase& test_case,
    const std::vector<NamedResult>& results)
{
    auto out = open_output(path);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"920\" height=\"670\" "
        << "viewBox=\"0 0 920 670\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"#f7f7f5\"/>\n";
    out << "<text x=\"70\" y=\"34\" font-family=\"Arial\" font-size=\"22\" "
        << "font-weight=\"700\">" << test_case.id << "</text>\n";
    out << "<text x=\"70\" y=\"56\" font-family=\"Arial\" font-size=\"13\" "
        << "fill=\"#555\">" << test_case.title << "</text>\n";

    double legend_x = 585.0;
    for (const NamedResult& named : results) {
        out << "<line x1=\"" << legend_x << "\" y1=\"36\" x2=\"" << (legend_x + 32.0)
            << "\" y2=\"36\" stroke=\"" << color_for(named.method)
            << "\" stroke-width=\"3\"" << dash_for(named.method) << "/>\n";
        out << "<text x=\"" << (legend_x + 40.0) << "\" y=\"41\" "
            << "font-family=\"Arial\" font-size=\"13\">" << swe::flux_name(named.method)
            << "</text>\n";
        legend_x += 105.0;
    }

    out << "<g font-family=\"Arial\" fill=\"#222\">\n";
    write_profile_panel(out, {72.0, 82.0, 800.0, 230.0}, results, "h", false);
    write_profile_panel(out, {72.0, 382.0, 800.0, 230.0}, results, "u", true);
    out << "<text x=\"452\" y=\"650\" font-size=\"14\">x</text>\n";
    out << "</g>\n</svg>\n";
}

void write_convergence_svg(
    const std::filesystem::path& path,
    const std::vector<ConvergenceSeries>& series)
{
    auto out = open_output(path);
    PlotRect rect{80.0, 70.0, 760.0, 420.0};

    double x_min = std::numeric_limits<double>::infinity();
    double x_max = -std::numeric_limits<double>::infinity();
    double y_min = std::numeric_limits<double>::infinity();
    double y_max = -std::numeric_limits<double>::infinity();
    for (const ConvergenceSeries& method_series : series) {
        for (const ConvergencePoint& point : method_series.points) {
            x_min = std::min(x_min, std::log10(static_cast<double>(point.cells)));
            x_max = std::max(x_max, std::log10(static_cast<double>(point.cells)));
            y_min = std::min(y_min, std::log10(point.h_l1));
            y_max = std::max(y_max, std::log10(point.h_l1));
        }
    }
    const auto [plot_y_min, plot_y_max] = expanded_range(y_min, y_max);

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"920\" height=\"560\" "
        << "viewBox=\"0 0 920 560\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"#f7f7f5\"/>\n";
    out << "<g font-family=\"Arial\" fill=\"#222\">\n";
    out << "<text x=\"80\" y=\"35\" font-size=\"22\" font-weight=\"700\">"
        << "Smooth test convergence</text>\n";
    write_svg_axes(out, rect, "log10 L1(h)", x_min, x_max, plot_y_min, plot_y_max);

    double legend_x = 590.0;
    for (const ConvergenceSeries& method_series : series) {
        out << "<polyline fill=\"none\" stroke=\"" << color_for(method_series.method)
            << "\" stroke-width=\"2.5\"" << dash_for(method_series.method)
            << " points=\"";
        for (const ConvergencePoint& point : method_series.points) {
            out << map_x(std::log10(static_cast<double>(point.cells)), rect, x_min, x_max)
                << ','
                << map_y(std::log10(point.h_l1), rect, plot_y_min, plot_y_max) << ' ';
        }
        out << "\"/>\n";
        for (const ConvergencePoint& point : method_series.points) {
            out << "<circle cx=\""
                << map_x(std::log10(static_cast<double>(point.cells)), rect, x_min, x_max)
                << "\" cy=\""
                << map_y(std::log10(point.h_l1), rect, plot_y_min, plot_y_max)
                << "\" r=\"4\" fill=\"" << color_for(method_series.method) << "\"/>\n";
        }

        out << "<line x1=\"" << legend_x << "\" y1=\"34\" x2=\"" << (legend_x + 32.0)
            << "\" y2=\"34\" stroke=\"" << color_for(method_series.method)
            << "\" stroke-width=\"3\"" << dash_for(method_series.method) << "/>\n";
        out << "<text x=\"" << (legend_x + 40.0) << "\" y=\"39\" font-size=\"13\">"
            << swe::flux_name(method_series.method) << "</text>\n";
        legend_x += 105.0;
    }

    out << "<text x=\"428\" y=\"530\" font-size=\"14\">log10 N</text>\n";
    out << "</g>\n</svg>\n";
}

void write_profile(const std::filesystem::path& path, const SolverResult& result)
{
    auto out = open_output(path);
    out << "x,h,u,hu\n";
    for (std::size_t i = 0; i < result.u.size(); ++i) {
        out << result.x[i] << ','
            << result.u[i].h << ','
            << swe::velocity(result.u[i]) << ','
            << result.u[i].hu << '\n';
    }
}

void write_mass_history(const std::filesystem::path& path, const SolverResult& result)
{
    auto out = open_output(path);
    out << "t,mass\n";
    for (const auto& sample : result.mass_history) {
        out << sample.t << ',' << sample.mass << '\n';
    }
}

SolverResult run_case(
    const TestCase& test_case,
    FluxKind flux,
    int cells,
    double cfl,
    int mass_samples = 250)
{
    SolverConfig config;
    config.cells = cells;
    config.x_min = test_case.x_min;
    config.x_max = test_case.x_max;
    config.t_final = test_case.t_final;
    config.cfl = cfl;
    config.flux = flux;
    config.boundary = test_case.boundary;
    config.max_mass_samples = mass_samples;
    return swe::solve(config, test_case.initial);
}

int nearest_reference_cell(const SolverResult& reference, double x)
{
    const double raw = (x - reference.x_min) / reference.dx;
    const int index = static_cast<int>(std::floor(raw));
    return std::clamp(index, 0, static_cast<int>(reference.u.size()) - 1);
}

ErrorMetrics compute_l1_error(const SolverResult& result, const SolverResult& reference)
{
    ErrorMetrics error;
    for (std::size_t i = 0; i < result.u.size(); ++i) {
        const int j = nearest_reference_cell(reference, result.x[i]);
        error.h_l1 += std::abs(result.u[i].h - reference.u[j].h) * result.dx;
        error.u_l1 +=
            std::abs(swe::velocity(result.u[i]) - swe::velocity(reference.u[j])) *
            result.dx;
    }
    return error;
}

int estimate_right_front_width_cells(const SolverResult& result)
{
    const int n = static_cast<int>(result.u.size());
    if (n < 8) {
        return 0;
    }

    const int tail_count = std::max(3, n / 20);
    double right_average = 0.0;
    for (int i = n - tail_count; i < n; ++i) {
        right_average += result.u[i].h;
    }
    right_average /= static_cast<double>(tail_count);

    double right_half_max = right_average;
    for (int i = 0; i < n; ++i) {
        if (result.x[i] > 0.0) {
            right_half_max = std::max(right_half_max, result.u[i].h);
        }
    }

    const double amplitude = right_half_max - right_average;
    if (amplitude < 1.0e-8) {
        return 0;
    }

    double max_jump = 0.0;
    for (int i = 0; i + 1 < n; ++i) {
        const double interface_x = 0.5 * (result.x[i] + result.x[i + 1]);
        if (interface_x > 0.0) {
            max_jump = std::max(max_jump, std::abs(result.u[i + 1].h - result.u[i].h));
        }
    }
    if (max_jump < 1.0e-12) {
        return 0;
    }

    return std::max(1, static_cast<int>(std::ceil(amplitude / max_jump)));
}

double last_mass(const SolverResult& result)
{
    return result.mass_history.empty() ? std::numeric_limits<double>::quiet_NaN()
                                       : result.mass_history.back().mass;
}

double first_mass(const SolverResult& result)
{
    return result.mass_history.empty() ? std::numeric_limits<double>::quiet_NaN()
                                       : result.mass_history.front().mass;
}

std::vector<TestCase> make_test_cases()
{
    return {
        {
            "dam_break",
            "Dam break h_L=2, h_R=1",
            -1.0,
            1.0,
            0.20,
            BoundaryKind::Outflow,
            [](double x) {
                return swe::make_state_from_h_u(x < 0.0 ? 2.0 : 1.0, 0.0);
            },
        },
        {
            "strong_break",
            "Strong dam break h_L=10, h_R=1",
            -1.0,
            1.0,
            0.08,
            BoundaryKind::Outflow,
            [](double x) {
                return swe::make_state_from_h_u(x < 0.0 ? 10.0 : 1.0, 0.0);
            },
        },
        {
            "smooth",
            "Smooth periodic wave",
            0.0,
            1.0,
            0.05,
            BoundaryKind::Periodic,
            [](double x) {
                const double h = 1.0 + 0.2 * std::sin(2.0 * kPi * x);
                return swe::make_state_from_h_u(h, 0.0);
            },
        },
    };
}

void write_summary(
    const std::filesystem::path& results_dir,
    const std::vector<TestCase>& test_cases,
    const std::vector<FluxKind>& methods)
{
    auto summary = open_output(results_dir / "summary.csv");
    summary << "case,title,method,N,t_final,cfl,steps,runtime_ms,stable,"
            << "positivity_fixes,min_h,mass0,mass_final,mass_abs_error,"
            << "l1_h_vs_hll_ref,l1_u_vs_hll_ref,right_front_width_cells,message\n";

    constexpr int n_profile = 400;
    constexpr int n_reference = 4096;
    constexpr double cfl = 0.45;

    for (const TestCase& test_case : test_cases) {
        std::cout << "Reference for " << test_case.id << "...\n";
        const SolverResult reference =
            run_case(test_case, FluxKind::Hll, n_reference, cfl, 10);
        std::vector<NamedResult> profile_plot_results;

        for (FluxKind method : methods) {
            std::cout << "  " << test_case.id << " / " << swe::flux_name(method) << "\n";
            const SolverResult result = run_case(test_case, method, n_profile, cfl);
            const ErrorMetrics error = compute_l1_error(result, reference);
            const std::string method_name = swe::flux_name(method);

            write_profile(
                results_dir / ("profile_" + test_case.id + "_" + method_name + ".csv"),
                result);
            write_mass_history(
                results_dir / ("mass_" + test_case.id + "_" + method_name + ".csv"),
                result);

            summary << test_case.id << ','
                    << '"' << test_case.title << '"' << ','
                    << method_name << ','
                    << n_profile << ','
                    << test_case.t_final << ','
                    << cfl << ','
                    << result.steps << ','
                    << result.runtime_ms << ','
                    << (result.stable ? "true" : "false") << ','
                    << result.positivity_fixes << ','
                    << result.min_h << ','
                    << first_mass(result) << ','
                    << last_mass(result) << ','
                    << std::abs(last_mass(result) - first_mass(result)) << ','
                    << error.h_l1 << ','
                    << error.u_l1 << ','
                    << (test_case.id == "smooth" ? 0 : estimate_right_front_width_cells(result)) << ','
                    << '"' << result.message << '"' << '\n';

            profile_plot_results.push_back({method, result});
        }

        write_profile_svg(
            results_dir / "plots" / ("profile_" + test_case.id + ".svg"),
            test_case,
            profile_plot_results);
    }
}

void write_convergence(
    const std::filesystem::path& results_dir,
    const TestCase& smooth_case,
    const std::vector<FluxKind>& methods)
{
    auto out = open_output(results_dir / "convergence_smooth.csv");
    out << "case,method,N,l1_h,l1_u,order_h,order_u,runtime_ms,mass_abs_error\n";

    const std::vector<int> grids{100, 200, 400, 800};
    constexpr int n_reference = 6400;
    constexpr double cfl = 0.45;
    std::vector<ConvergenceSeries> plot_series;

    for (FluxKind method : methods) {
        const std::string method_name = swe::flux_name(method);
        std::cout << "Convergence reference / " << method_name << "...\n";
        const SolverResult reference = run_case(smooth_case, method, n_reference, cfl, 10);

        double previous_h = std::numeric_limits<double>::quiet_NaN();
        double previous_u = std::numeric_limits<double>::quiet_NaN();
        ConvergenceSeries current_series;
        current_series.method = method;

        for (int n : grids) {
            const SolverResult result = run_case(smooth_case, method, n, cfl, 20);
            const ErrorMetrics error = compute_l1_error(result, reference);

            double order_h = std::numeric_limits<double>::quiet_NaN();
            double order_u = std::numeric_limits<double>::quiet_NaN();
            if (std::isfinite(previous_h) && error.h_l1 > 0.0) {
                order_h = std::log(previous_h / error.h_l1) / std::log(2.0);
            }
            if (std::isfinite(previous_u) && error.u_l1 > 0.0) {
                order_u = std::log(previous_u / error.u_l1) / std::log(2.0);
            }

            out << smooth_case.id << ','
                << method_name << ','
                << n << ','
                << error.h_l1 << ','
                << error.u_l1 << ','
                << order_h << ','
                << order_u << ','
                << result.runtime_ms << ','
                << std::abs(last_mass(result) - first_mass(result)) << '\n';

            previous_h = error.h_l1;
            previous_u = error.u_l1;
            current_series.points.push_back({n, error.h_l1});
        }
        plot_series.push_back(current_series);
    }

    write_convergence_svg(results_dir / "plots" / "convergence_smooth.svg", plot_series);
}

void write_stability(
    const std::filesystem::path& results_dir,
    const TestCase& dam_break,
    const std::vector<FluxKind>& methods)
{
    auto out = open_output(results_dir / "stability_cfl.csv");
    out << "case,method,cfl,N,steps,stable,positivity_fixes,min_h,mass_abs_error,message\n";

    const std::vector<double> cfl_values{0.2, 0.5, 0.9, 1.1};
    constexpr int n = 400;

    for (FluxKind method : methods) {
        for (double cfl : cfl_values) {
            const SolverResult result = run_case(dam_break, method, n, cfl, 20);
            out << dam_break.id << ','
                << swe::flux_name(method) << ','
                << cfl << ','
                << n << ','
                << result.steps << ','
                << (result.stable ? "true" : "false") << ','
                << result.positivity_fixes << ','
                << result.min_h << ','
                << std::abs(last_mass(result) - first_mass(result)) << ','
                << '"' << result.message << '"' << '\n';
        }
    }
}

} // namespace

int main()
{
    try {
        const std::filesystem::path results_dir = "results";
        std::filesystem::create_directories(results_dir);
        std::filesystem::create_directories(results_dir / "plots");

        const std::vector<FluxKind> methods{
            FluxKind::Rusanov,
            FluxKind::Hll,
            FluxKind::Roe,
        };
        const std::vector<TestCase> test_cases = make_test_cases();

        write_summary(results_dir, test_cases, methods);

        const auto smooth_it = std::find_if(
            test_cases.begin(),
            test_cases.end(),
            [](const TestCase& test_case) { return test_case.id == "smooth"; });
        if (smooth_it != test_cases.end()) {
            write_convergence(results_dir, *smooth_it, methods);
        }

        write_stability(results_dir, test_cases.front(), methods);

        std::cout << "Done. CSV and SVG files are in " << results_dir.string() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
