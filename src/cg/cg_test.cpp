// cg_test.cpp — Test and comparison of Standard CG and Preconditioned CG
//
// Matrix source is selected at runtime:
//   ./cg_test                          — synthetic tridiagonal SPD, N=1000
//   ./cg_test --mm-file path/to/A.mtx  — Matrix Market file
//
// Pass criterion: ||A*x - b|| / ||b|| < 1e-8

#include "cg.h"
#include "ilu.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <string>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static constexpr double TOL = 1e-11;
static constexpr double PASS_TOL = 1e-10;
static constexpr int MAXITER = 10000;

// ---------------------------------------------------------------------------
// Synthetic tridiagonal SPD matvec (1D Laplacian)
// ---------------------------------------------------------------------------
static void tridiag_matvec(int n, const linalg::Vec &x, linalg::Vec &y)
{
  y.assign(n, 0.0);
  for (int i = 0; i < n; ++i)
  {
    y[i] += 2.0 * x[i];
    if (i > 0)
      y[i] -= 1.0 * x[i - 1];
    if (i + 1 < n)
      y[i] -= 1.0 * x[i + 1];
  }
}

// Build CSR for synthetic matrix
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
    A.values[k++] = 2.0;
    if (i < n - 1)
    {
      A.col_idx[k] = i + 1;
      A.values[k++] = -1.0;
    }
  }
  return A;
}

// ---------------------------------------------------------------------------
// Simple Diagonal (Jacobi) Preconditioner (M = diag(A))
// ---------------------------------------------------------------------------
cg::Precond make_jacobi_precond(const CSRMatrix &A)
{
  int n = A.rows;
  linalg::Vec inv_diag(n, 1.0);

  // Извлекаем диагональ
  for (int i = 0; i < n; ++i)
  {
    for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
    {
      if (A.col_idx[k] == i)
      {
        // Предотвращение деления на ноль
        inv_diag[i] = (A.values[k] != 0.0) ? (1.0 / A.values[k]) : 1.0;
        break;
      }
    }
  }

  // Возвращаем функцию (лямбду), выполняющую z = M^-1 * r
  return [inv_diag](const linalg::Vec &r, linalg::Vec &z)
  {
    z.resize(r.size());
    for (size_t i = 0; i < r.size(); ++i)
    {
      z[i] = r[i] * inv_diag[i];
    }
  };
}

// ---------------------------------------------------------------------------
// Residual check
// ---------------------------------------------------------------------------
static double check_residual(const cg::Matvec &mv,
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
      std::cerr << "Usage: cg_test [--mm-file <path>]\n";
      return 1;
    }
  }

  int N = 0;
  linalg::Vec b;
  cg::Matvec mv;
  CSRMatrix csr;

  ilu::ScaledILU0 silu_f;
  bool silu_ok = false;
  cg::Precond jacobi_precond;

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

    jacobi_precond = make_jacobi_precond(csr);

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
    N = 1000;
    csr = make_tridiag_csr(N);
    mv = [&](const linalg::Vec &x, linalg::Vec &y)
    { tridiag_matvec(N, x, y); };
    linalg::Vec x_exact(N, 1.0);
    b.resize(N);
    mv(x_exact, b);
    std::cout << "Using synthetic tridiagonal SPD matrix (1D Laplace), N=" << N << "\n";

    jacobi_precond = make_jacobi_precond(csr);

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

  std::cout << "\nParameters: tol=" << TOL << ", max_iter=" << MAXITER << "\n\n";
  std::cout << std::left << std::setw(16) << "Solver"
            << std::right << std::setw(8) << "MV-ops"
            << std::setw(16) << "rel_res"
            << std::setw(14) << "wall time" << "   status\n";
  std::cout << std::string(62, '-') << "\n";

  bool all_ok = true;

  // Standard CG (без предобуславливания)
  {
    linalg::Vec x(N, 0.0);
    auto t0 = Clock::now();
    auto res = cg::solve(mv, b, x, TOL, MAXITER);

    // В CG количество итераций равно количеству MV-операций + 1
    int mv_ops = res.iters + 1;

    double err = check_residual(mv, x, b);
    bool ok = err < PASS_TOL;
    print_row("CG (Standard)", mv_ops, err, ms_since(t0), ok);
    if (!ok)
    {
      std::cerr << "[FAIL] CG: " << err << "\n";
      all_ok = false;
    }
  }

  // Preconditioned CG (Jacobi / Диагональный)
  {
    linalg::Vec x(N, 0.0);
    auto t0 = Clock::now();
    auto res = cg::solve(mv, b, x, TOL, MAXITER, jacobi_precond);

    int mv_ops = res.iters + 1;

    double err = check_residual(mv, x, b);
    bool ok = err < PASS_TOL;
    print_row("CG+Jacobi", mv_ops, err, ms_since(t0), ok);
    if (!ok)
    {
      std::cerr << "[FAIL] CG+Jacobi: " << err << "\n";
      all_ok = false;
    }
  }

  // Preconditioned CG (sILU(0))
  if (silu_ok)
  {
    auto ilu_precond = ilu::make_scaled_precond(silu_f);
    linalg::Vec x(N, 0.0);
    auto t0 = Clock::now();
    auto res = cg::solve(mv, b, x, TOL, MAXITER, ilu_precond);

    int mv_ops = res.iters + 1;

    double err = check_residual(mv, x, b);
    bool ok = err < PASS_TOL;
    print_row("CG+sILU0", mv_ops, err, ms_since(t0), ok);
    if (!ok)
    {
      std::cerr << "[FAIL] CG+sILU0: " << err << "\n";
      all_ok = false;
    }
  }
  else
  {
    std::cout << std::left << std::setw(16) << "CG+sILU0"
              << "  (skipped — factorization failed)\n";
  }

  std::cout << "\n";
  if (all_ok)
    std::cout << "All correctness checks passed.\n";
  return all_ok ? 0 : 1;
}
