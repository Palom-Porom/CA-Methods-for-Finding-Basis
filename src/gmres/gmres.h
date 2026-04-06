#pragma once
// gmres.h — Classical restarted GMRES(m)
// Arnoldi with modified Gram-Schmidt + Givens rotations QR
//
// Supports optional right preconditioning:
//   solve  A * M^{-1} * u = b,  x = M^{-1} * u
// Pass a Precond functor (e.g. ilu::make_precond) as the last argument.
// If precond == nullptr the solver behaves exactly as before.

#include "linalg.h"
#include "matrix_market.h"
#include <functional>
#include <cmath>
#include <vector>
#include <cassert>
#include <stdexcept>

namespace gmres {

using Vec  = linalg::Vec;
using Mat  = linalg::Mat;

using Matvec  = std::function<void(const Vec&, Vec&)>;
using Precond = std::function<void(const Vec&, Vec&)>;  // solves M*z = r

struct Result {
    int    iters;
    double rel_res;
    bool   converged;
};

// ---------------------------------------------------------------------------
// Classical right-preconditioned GMRES(restart)
//
// If precond == nullptr: unpreconditioned (original behaviour preserved)
// If precond != nullptr: right-preconditioned
//   — Krylov space built with (A * M^{-1})
//   — solution recovered as x += Z * y  where Z[:,j] = M^{-1} * v_j
// ---------------------------------------------------------------------------
inline Result solve(
    const Matvec&  matvec,
    const Vec&     b,
    Vec&           x,
    int            restart  = 30,
    double         tol      = 1e-10,
    int            max_iter = 1000,
    const Precond& precond  = nullptr)
{
    const int n = static_cast<int>(b.size());
    assert(static_cast<int>(x.size()) == n);

    const bool use_precond = (precond != nullptr);

    Vec r(n), Ax(n);
    matvec(x, Ax);
    linalg::copy(b, r);
    linalg::axpy(-1.0, Ax, r);

    double r0_norm = linalg::nrm2(r);
    if (r0_norm == 0.0) return {0, 0.0, true};

    int    total_mv  = 1;
    double rel_res   = 1.0;
    bool   converged = false;

    Mat V = linalg::make_mat(n, restart + 1);
    Mat H = linalg::make_mat(restart + 1, restart);
    // Z[:,j] = M^{-1} * v_j — only allocated when preconditioner is active
    Mat Z = use_precond ? linalg::make_mat(n, restart) : Mat{};
    std::vector<double> cs(restart), sn(restart);
    Vec g(restart + 1);
    Vec tmp(n);

    while (total_mv < max_iter && !converged) {
        double beta = linalg::nrm2(r);
        rel_res = beta / r0_norm;
        if (rel_res < tol) { converged = true; break; }

        double* v0 = linalg::col(V, n, 0);
        linalg::copy(n, r.data(), v0);
        linalg::scal(n, 1.0 / beta, v0);

        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;
        std::fill(H.begin(), H.end(), 0.0);

        int j = 0;
        for (; j < restart && total_mv < max_iter; ++j) {
            Vec w(n);

            if (use_precond) {
                // z_j = M^{-1} * v_j,  w = A * z_j
                Vec vj(linalg::col(V, n, j), linalg::col(V, n, j) + n);
                precond(vj, tmp);
                linalg::copy(n, tmp.data(), linalg::col(Z, n, j));
                matvec(tmp, w);
            } else {
                Vec vj(linalg::col(V, n, j), linalg::col(V, n, j) + n);
                matvec(vj, w);
            }
            ++total_mv;

            // Modified Gram-Schmidt orthogonalisation
            for (int i = 0; i <= j; ++i) {
                const double* vi = linalg::col(V, n, i);
                double h = linalg::dot(n, w.data(), vi);
                H[j * (restart + 1) + i] = h;
                linalg::axpy(n, -h, vi, w.data());
            }
            double h_next = linalg::nrm2(w);
            H[j * (restart + 1) + j + 1] = h_next;

            if (h_next > 0.0) {
                double* vj1 = linalg::col(V, n, j + 1);
                linalg::copy(n, w.data(), vj1);
                linalg::scal(n, 1.0 / h_next, vj1);
            }

            // Apply previous Givens rotations
            for (int i = 0; i < j; ++i) {
                double h_i   = H[j * (restart + 1) + i];
                double h_ip1 = H[j * (restart + 1) + i + 1];
                H[j * (restart + 1) + i]     =  cs[i] * h_i + sn[i] * h_ip1;
                H[j * (restart + 1) + i + 1] = -sn[i] * h_i + cs[i] * h_ip1;
            }

            // New Givens rotation to zero H[j+1, j]
            double h_jj   = H[j * (restart + 1) + j];
            double h_jp1j = H[j * (restart + 1) + j + 1];
            double denom  = std::hypot(h_jj, h_jp1j);
            if (denom < 1e-300) { cs[j] = 1.0; sn[j] = 0.0; }
            else { cs[j] = h_jj / denom; sn[j] = h_jp1j / denom; }

            H[j * (restart + 1) + j]     =  cs[j] * h_jj + sn[j] * h_jp1j;
            H[j * (restart + 1) + j + 1] = 0.0;

            g[j + 1] = -sn[j] * g[j];
            g[j]     =  cs[j] * g[j];

            rel_res = std::abs(g[j + 1]) / r0_norm;
            if (rel_res < tol) { j++; converged = true; break; }
        }

        // Back-substitution
        int m = j;
        Vec y(m);
        for (int i = m - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int k = i + 1; k < m; ++k)
                y[i] -= H[k * (restart + 1) + i] * y[k];
            y[i] /= H[i * (restart + 1) + i];
        }

        // x += Z * y  (preconditioned)  or  x += V * y  (plain)
        const Mat& basis = use_precond ? Z : V;
        for (int i = 0; i < m; ++i)
            linalg::axpy(n, y[i], linalg::col(basis, n, i), x.data());

        if (converged) break;

        // Recompute true residual for next restart cycle
        matvec(x, Ax); ++total_mv;
        linalg::copy(b, r);
        linalg::axpy(-1.0, Ax, r);
    }

    // Final accurate residual
    matvec(x, Ax);
    linalg::copy(b, r);
    linalg::axpy(-1.0, Ax, r);
    rel_res = linalg::nrm2(r) / r0_norm;

    return {total_mv, rel_res, rel_res < tol};
}

// ---------------------------------------------------------------------------
// CSRMatrix convenience overload
// ---------------------------------------------------------------------------
inline Result solve(
    const CSRMatrix& A,
    const Vec&       b,
    Vec&             x,
    int              restart  = 30,
    double           tol      = 1e-10,
    int              max_iter = 1000,
    const Precond&   precond  = nullptr)
{
    if (A.rows != A.cols)
        throw std::runtime_error(
            "gmres::solve: matrix must be square ("
            + std::to_string(A.rows) + "x" + std::to_string(A.cols) + ")");
    if (static_cast<int>(b.size()) != A.rows)
        throw std::runtime_error("gmres::solve: b size mismatch");
    if (static_cast<int>(x.size()) != A.cols)
        x.assign(A.cols, 0.0);

    return solve(spmv_matvec(A), b, x, restart, tol, max_iter, precond);
}

} // namespace gmres