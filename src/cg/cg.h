#pragma once
// cg.h — Classical and Preconditioned Conjugate Gradient (CG) solver
//
// Supports optional preconditioner:
//   solve  M^{-1} A x = M^{-1} b
// Pass a Precond functor as the last argument.
// If precond == nullptr, the standard CG algorithm is used.
//
// NOTE: Matrix A must be Symmetric Positive-Definite (SPD).
// If a preconditioner M is used, it must also be SPD.

#include "linalg.h"
#include "matrix_market.h" // Подразумевается, что тут определен CSRMatrix и spmv_matvec
#include <functional>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace cg
{

  using Vec = linalg::Vec;
  using Matvec = std::function<void(const Vec &, Vec &)>;
  using Precond = std::function<void(const Vec &, Vec &)>; // solves M*z = r

  struct Result
  {
    int iters;
    double rel_res;
    bool converged;
  };

  // ---------------------------------------------------------------------------
  // Preconditioned Conjugate Gradient
  // ---------------------------------------------------------------------------
  inline Result solve(
      const Matvec &matvec,
      const Vec &b,
      Vec &x,
      double tol = 1e-10,
      int max_iter = 1000,
      const Precond &precond = nullptr)
  {
    const int n = static_cast<int>(b.size());
    if (static_cast<int>(x.size()) != n)
    {
      x.assign(n, 0.0);
    }

    const bool use_precond = (precond != nullptr);

    Vec r(n), p(n), Ap(n), z;
    if (use_precond)
    {
      z.resize(n);
    }

    // Шаг 1: Инициализация
    // r = b - A*x
    matvec(x, Ap);
    linalg::copy(b, r);
    linalg::axpy(-1.0, Ap, r);

    double r0_norm = linalg::nrm2(r);
    if (r0_norm == 0.0)
      return {0, 0.0, true};

    // z = M^-1 * r (если есть предобуславливатель)
    if (use_precond)
    {
      precond(r, z);
      linalg::copy(z, p);
    }
    else
    {
      linalg::copy(r, p);
    }

    double rz_old = use_precond ? linalg::dot(r, z) : linalg::dot(r, r);
    double rel_res = 1.0;

    for (int i = 0; i < max_iter; ++i)
    {
      // Шаг 2: Умножение матрицы на вектор
      matvec(p, Ap);

      // Шаг 3: Вычисление длины шага alpha
      double pAp = linalg::dot(p, Ap);

      // Защита от деления на ноль (матрица не SPD)
      if (std::abs(pAp) < 1e-300)
      {
        std::cerr << "[WARN] cg::solve - p^T A p is near zero. Matrix might not be SPD.\n";
        return {i, linalg::nrm2(r) / r0_norm, false};
      }

      double alpha = rz_old / pAp;

      // Шаг 4: Обновление решения и невязки
      linalg::axpy(alpha, p, x);   // x = x + alpha * p
      linalg::axpy(-alpha, Ap, r); // r = r - alpha * Ap

      // Шаг 5: Проверка сходимости
      double r_norm = linalg::nrm2(r);
      rel_res = r_norm / r0_norm;

      if (rel_res < tol)
      {
        return {i + 1, rel_res, true};
      }

      // Шаг 6: Вычисление коэффициента сопряженности beta и нового направления
      if (use_precond)
      {
        precond(r, z);
      }

      double rz_new = use_precond ? linalg::dot(r, z) : r_norm * r_norm;
      double beta = rz_new / rz_old;

      // p = z + beta * p (или p = r + beta * p, если нет предобуславливателя)
      linalg::scal(beta, p); // p = beta * p
      if (use_precond)
      {
        linalg::axpy(1.0, z, p); // p = beta * p + z
      }
      else
      {
        linalg::axpy(1.0, r, p); // p = beta * p + r
      }

      rz_old = rz_new;
    }

    return {max_iter, rel_res, false};
  }

  // ---------------------------------------------------------------------------
  // CSRMatrix convenience overload
  // ---------------------------------------------------------------------------
  inline Result solve(
      const CSRMatrix &A,
      const Vec &b,
      Vec &x,
      double tol = 1e-10,
      int max_iter = 1000,
      const Precond &precond = nullptr)
  {
    if (A.rows != A.cols)
      throw std::runtime_error(
          "cg::solve: matrix must be square (" + std::to_string(A.rows) + "x" + std::to_string(A.cols) + ")");
    if (static_cast<int>(b.size()) != A.rows)
      throw std::runtime_error("cg::solve: b size mismatch");
    if (static_cast<int>(x.size()) != A.cols)
      x.assign(A.cols, 0.0);

    return solve(spmv_matvec(A), b, x, tol, max_iter, precond);
  }

} // namespace cg
