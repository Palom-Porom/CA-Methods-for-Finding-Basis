// gmres_test.cpp — Test and comparison of GMRES vs CA-GMRES
//
// Problem: tridiagonal SPD system of size N=100
//   A[i,i]   =  4
//   A[i,i+1] = -1
//   A[i+1,i] = -1
// RHS b chosen so that the exact solution is x* = [1, 1, ..., 1]^T
//
// Both solvers are run with identical parameters and their performance
// (iterations, final residual, wall time) is compared.
//
// Pass criterion: ||A*x - b|| / ||b|| < 1e-10

#include "gmres.h"
#include "ca_gmres.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cassert>

// ---------------------------------------------------------------------------
// Build tridiagonal SPD matrix (column-major)
// ---------------------------------------------------------------------------
static linalg::Mat make_tridiag(int n) {
    linalg::Mat A(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        A[i * n + i] = 4.0;
        if (i + 1 < n) {
            A[(i + 1) * n + i] = -1.0;   // sub-diagonal
            A[i * n + (i + 1)] = -1.0;   // super-diagonal
        }
    }
    return A;
}

// ---------------------------------------------------------------------------
// Sparse tridiagonal matvec (avoids storing dense n*n matrix)
// ---------------------------------------------------------------------------
static void tridiag_matvec(int n, const linalg::Vec& x, linalg::Vec& y) {
    y.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        y[i] += 4.0 * x[i];
        if (i > 0)     y[i] -= x[i - 1];
        if (i + 1 < n) y[i] -= x[i + 1];
    }
}

// ---------------------------------------------------------------------------
// Accuracy check: returns ||A*x - b|| / ||b||
// ---------------------------------------------------------------------------
static double check_residual(int n, const linalg::Vec& x, const linalg::Vec& b) {
    linalg::Vec Ax(n);
    tridiag_matvec(n, x, Ax);
    linalg::axpy(-1.0, b, Ax);          // Ax <- Ax - b
    return linalg::nrm2(Ax) / linalg::nrm2(b);
}

// ---------------------------------------------------------------------------
// Pretty-print result row
// ---------------------------------------------------------------------------
static void print_row(const char* name, int iters, double rel_res,
                      double wall_ms, bool pass) {
    std::cout << std::left  << std::setw(14) << name
              << std::right << std::setw(8)  << iters
              << std::setw(16) << std::scientific << std::setprecision(4) << rel_res
              << std::setw(12) << std::fixed      << std::setprecision(3) << wall_ms << " ms"
              << "   " << (pass ? "PASS" : "FAIL") << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    constexpr int    N       = 1e7;
    constexpr double TOL     = 1e-10;
    constexpr int    RESTART = 64;    // Krylov dimension per cycle
    constexpr int    S       =  16;    // CA-GMRES s-step size
    constexpr int    MAXITER = 2000;

    // Build RHS: b = A * x_exact,  x_exact = [1,...,1]
    linalg::Vec x_exact(N, 1.0);
    linalg::Vec b(N);
    tridiag_matvec(N, x_exact, b);

    auto mv = [&](const linalg::Vec& x, linalg::Vec& y) {
        tridiag_matvec(N, x, y);
    };

    // -----------------------------------------------------------------------
    // Run classical GMRES
    // -----------------------------------------------------------------------
    linalg::Vec x_gmres(N, 0.0);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto res_gmres = gmres::solve(mv, b, x_gmres, RESTART, TOL, MAXITER);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_gmres = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double err_gmres = check_residual(N, x_gmres, b);

    // -----------------------------------------------------------------------
    // Run CA-GMRES
    // -----------------------------------------------------------------------
    linalg::Vec x_cagmres(N, 0.0);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto res_ca = ca_gmres::solve(mv, b, x_cagmres, S, RESTART, TOL, MAXITER);
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_ca = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double err_ca = check_residual(N, x_cagmres, b);

    // -----------------------------------------------------------------------
    // Report
    // -----------------------------------------------------------------------
    std::cout << "\n";
    std::cout << "Problem : tridiagonal SPD,  N=" << N
              << ",  tol=" << TOL
              << ",  restart=" << RESTART
              << ",  s=" << S << "\n\n";

    std::cout << std::left  << std::setw(14) << "Solver"
              << std::right << std::setw(8)  << "MV-ops"
              << std::setw(16) << "rel_res"
              << std::setw(14) << "wall time"
              << "   status\n";
    std::cout << std::string(60, '-') << "\n";

    constexpr double PASS_TOL = 1e-10;
    print_row("GMRES",    res_gmres.iters, err_gmres, ms_gmres, err_gmres < PASS_TOL);
    print_row("CA-GMRES", res_ca.iters,    err_ca,    ms_ca,    err_ca    < PASS_TOL);

    std::cout << "\n";

    // -----------------------------------------------------------------------
    // Assertions (exit non-zero on failure)
    // -----------------------------------------------------------------------
    bool ok = true;
    if (err_gmres >= PASS_TOL) {
        std::cerr << "[FAIL] GMRES: ||Ax-b||/||b|| = " << err_gmres
                  << " >= " << PASS_TOL << "\n";
        ok = false;
    }
    if (err_ca >= PASS_TOL) {
        std::cerr << "[FAIL] CA-GMRES: ||Ax-b||/||b|| = " << err_ca
                  << " >= " << PASS_TOL << "\n";
        ok = false;
    }
    if (ok) {
        std::cout << "All correctness checks passed.\n";
    }

    return ok ? 0 : 1;
}
