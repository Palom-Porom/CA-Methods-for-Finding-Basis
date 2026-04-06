// gmres_test.cpp — Test and comparison of GMRES, CA-GMRES, GMRES+ILU(0)
//
// Matrix source is selected at runtime:
//   ./gmres_test                          — synthetic tridiagonal SPD, N=100
//   ./gmres_test --mm-file path/to/A.mtx  — Matrix Market file
//
// Use build.sh --mm-file <path> to pass an absolute path automatically.
//
// Pass criterion: ||A*x - b|| / ||b|| < 1e-10

#include "gmres.h"
#include "ca_gmres.h"
#include "ilu.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <string>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static constexpr double TOL = 1e-10;
static constexpr double PASS_TOL = 1e-10;
static constexpr int RESTART = 64;
static constexpr int S = 16;
static constexpr int MAXITER = 10000;

// ---------------------------------------------------------------------------
// Synthetic tridiagonal SPD matvec
// ---------------------------------------------------------------------------
static void tridiag_matvec(int n, const linalg::Vec &x, linalg::Vec &y)
{
    y.assign(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        y[i] += 4.0 * x[i];
        if (i > 0)
            y[i] -= x[i - 1];
        if (i + 1 < n)
            y[i] -= x[i + 1];
    }
}

// Build CSR for synthetic matrix (needed to construct ILU0)
static CSRMatrix make_tridiag_csr(int n)
{
    CSRMatrix A;
    A.rows = A.cols = n;
    A.row_ptr.resize(n + 1, 0);
    for (int i = 0; i < n; ++i)
        A.row_ptr[i + 1] = A.row_ptr[i] + (i == 0 || i == n - 1 ? 2 : 3);
    A.nnz = A.row_ptr[n];
    A.values.resize(A.nnz);
    A.col_idx.resize(A.nnz);
    for (int i = 0; i < n; ++i)
    {
        int k = A.row_ptr[i];
        if (i > 0)
        {
            A.col_idx[k] = i - 1;
            A.values[k++] = -1.0;
        }
        A.col_idx[k] = i;
        A.values[k++] = 4.0;
        if (i < n - 1)
        {
            A.col_idx[k] = i + 1;
            A.values[k++] = -1.0;
        }
    }
    return A;
}

// ---------------------------------------------------------------------------
// Residual check
// ---------------------------------------------------------------------------
static double check_residual(const gmres::Matvec &mv,
                             const linalg::Vec &x,
                             const linalg::Vec &b)
{
    linalg::Vec Ax(x.size());
    mv(x, Ax);
    linalg::axpy(-1.0, b, Ax);
    return linalg::nrm2(Ax) / linalg::nrm2(b);
}

using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static void print_row(const char *name, int iters, double rel_res,
                      double wall_ms, bool pass)
{
    std::cout << std::left << std::setw(16) << name
              << std::right << std::setw(8) << iters
              << std::setw(16) << std::scientific << std::setprecision(4) << rel_res
              << std::setw(12) << std::fixed << std::setprecision(3) << wall_ms << " ms"
              << "   " << (pass ? "PASS" : "FAIL") << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{

    std::string mm_file;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--mm-file")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "[ERROR] --mm-file requires a path\n";
                return 1;
            }
            mm_file = argv[++i];
        }
        else
        {
            std::cerr << "[ERROR] Unknown argument: " << arg << "\n";
            std::cerr << "Usage: gmres_test [--mm-file <path>]\n";
            return 1;
        }
    }

    int N = 0;
    linalg::Vec b;
    gmres::Matvec mv;
    CSRMatrix csr;
    ilu::ScaledILU0 silu_f;
    bool silu_ok = false;

    if (!mm_file.empty())
    {
        std::cout << "Loading matrix from: " << mm_file << "\n";
        try
        {
            csr = load_matrix_market(mm_file);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] " << e.what() << "\n";
            return 1;
        }

        if (csr.rows != csr.cols)
        {
            std::cerr << "[ERROR] Matrix not square\n";
            return 1;
        }
        N = csr.rows;
        mv = spmv_matvec(csr);
        linalg::Vec x_exact(N, 1.0);
        b.resize(N);
        mv(x_exact, b);
        std::cout << "Matrix: " << N << "x" << N << ", nnz=" << csr.nnz << "\n";

        try
        {
            auto t = Clock::now();
            silu_f = ilu::build_scaled_ilu0(csr);
            silu_ok = true;
            std::cout << "sILU(0) factored in " << ms_since(t) << " ms\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "[WARN] sILU(0) failed: " << e.what() << " — row skipped\n";
        }
    }
    else
    {
        N = 100;
        csr = make_tridiag_csr(N);
        mv = [&](const linalg::Vec &x, linalg::Vec &y)
        { tridiag_matvec(N, x, y); };
        linalg::Vec x_exact(N, 1.0);
        b.resize(N);
        mv(x_exact, b);
        std::cout << "Using synthetic tridiagonal SPD matrix, N=" << N << "\n";
        try
        {
            silu_f = ilu::build_scaled_ilu0(csr);
            silu_ok = true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[WARN] sILU(0) failed: " << e.what() << "\n";
        }
    }

    std::cout << "\nParameters: tol=" << TOL << ", restart=" << RESTART
              << ", s=" << S << ", max_iter=" << MAXITER << "\n\n";
    std::cout << std::left << std::setw(16) << "Solver"
              << std::right << std::setw(8) << "MV-ops"
              << std::setw(16) << "rel_res"
              << std::setw(14) << "wall time" << "   status\n";
    std::cout << std::string(62, '-') << "\n";

    bool all_ok = true;

    // GMRES
    {
        linalg::Vec x(N, 0.0);
        auto t0 = Clock::now();
        auto res = gmres::solve(mv, b, x, RESTART, TOL, MAXITER);
        double err = check_residual(mv, x, b);
        bool ok = err < PASS_TOL;
        print_row("GMRES", res.iters, err, ms_since(t0), ok);
        if (!ok)
        {
            std::cerr << "[FAIL] GMRES: " << err << "\n";
            all_ok = false;
        }
    }

    // CA-GMRES
    {
        linalg::Vec x(N, 0.0);
        auto t0 = Clock::now();
        auto res = ca_gmres::solve(mv, b, x, S, RESTART, TOL, MAXITER);
        double err = check_residual(mv, x, b);
        bool ok = err < PASS_TOL;
        print_row("CA-GMRES", res.iters, err, ms_since(t0), ok);
        if (!ok)
        {
            std::cerr << "[FAIL] CA-GMRES: " << err << "\n";
            all_ok = false;
        }
    }

    // GMRES + sILU(0)
    if (silu_ok)
    {
        auto precond = ilu::make_scaled_precond(silu_f);
        linalg::Vec x(N, 0.0);
        auto t0 = Clock::now();
        auto res = gmres::solve(mv, b, x, RESTART, TOL, MAXITER, precond);
        double err = check_residual(mv, x, b);
        bool ok = err < PASS_TOL;
        print_row("GMRES+sILU0", res.iters, err, ms_since(t0), ok);
        if (!ok)
        {
            std::cerr << "[FAIL] GMRES+sILU0: " << err << "\n";
            all_ok = false;
        }
    }
    else
    {
        std::cout << std::left << std::setw(16) << "GMRES+sILU0"
                  << "  (skipped — factorization failed)\n";
    }

    // CA-GMRES + sILU(0)
    if (silu_ok)
    {
        auto precond = ilu::make_scaled_precond(silu_f);
        linalg::Vec x(N, 0.0);
        auto t0 = Clock::now();
        auto res = ca_gmres::solve(mv, b, x, S, RESTART, TOL, MAXITER, precond);
        double err = check_residual(mv, x, b);
        bool ok = err < PASS_TOL;
        print_row("CA-GMRES+sILU0", res.iters, err, ms_since(t0), ok);
        if (!ok)
        {
            std::cerr << "[FAIL] CA-GMRES+sILU0: " << err << "\n";
            all_ok = false;
        }
    }
    else
    {
        std::cout << std::left << std::setw(16) << "CA-GMRES+sILU0"
                  << "  (skipped — factorization failed)\n";
    }

    std::cout << "\n";
    if (all_ok)
        std::cout << "All correctness checks passed.\n";
    return all_ok ? 0 : 1;
}
