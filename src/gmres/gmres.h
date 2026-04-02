#pragma once
// gmres.h — Classical restarted GMRES(m)
// Arnoldi with modified Gram-Schmidt + Givens rotations QR

#include "linalg.h"
#include <functional>
#include <cmath>
#include <vector>
#include <cassert>

namespace gmres {

using Vec  = linalg::Vec;
using Mat  = linalg::Mat;

// Matvec functor type: y <- A*x
using Matvec = std::function<void(const Vec& x, Vec& y)>;

struct Result {
    int    iters;       // total matrix-vector products
    double rel_res;     // ||r_final|| / ||r_0||
    bool   converged;
};

// ---------------------------------------------------------------------------
// Classical GMRES(restart)
// ---------------------------------------------------------------------------
inline Result solve(
    const Matvec& matvec,
    const Vec&    b,
    Vec&          x,           // initial guess; overwritten with solution
    int           restart = 30,
    double        tol     = 1e-10,
    int           max_iter = 1000)
{
    const int n = static_cast<int>(b.size());
    assert(static_cast<int>(x.size()) == n);

    // r0 = b - A*x
    Vec r(n), Ax(n);
    matvec(x, Ax);
    linalg::copy(b, r);
    linalg::axpy(-1.0, Ax, r);

    double r0_norm = linalg::nrm2(r);
    if (r0_norm == 0.0) return {0, 0.0, true};

    int total_mv = 1;
    double rel_res = 1.0;
    bool converged = false;

    // Krylov basis V: columns 0..m  (n x (restart+1)) column-major
    Mat V = linalg::make_mat(n, restart + 1);
    // Upper Hessenberg H: (restart+1) x restart column-major
    Mat H = linalg::make_mat(restart + 1, restart);
    // Givens cos/sin
    std::vector<double> cs(restart), sn(restart);
    // RHS for least-squares in Krylov coords
    Vec g(restart + 1);

    while (total_mv < max_iter && !converged) {
        // --- Initialise cycle ---
        // v_0 = r / ||r||
        double beta = linalg::nrm2(r);
        rel_res = beta / r0_norm;
        if (rel_res < tol) { converged = true; break; }

        double inv_beta = 1.0 / beta;
        double* v0 = linalg::col(V, n, 0);
        linalg::copy(n, r.data(), v0);
        linalg::scal(n, inv_beta, v0);

        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;
        std::fill(H.begin(), H.end(), 0.0);

        int j = 0; // current Arnoldi step
        for (; j < restart && total_mv < max_iter; ++j) {
            // w = A * v_j
            Vec w(n);
            {
                Vec vj(linalg::col(V, n, j), linalg::col(V, n, j) + n);
                matvec(vj, w);
            }
            ++total_mv;

            // Modified Gram-Schmidt
            for (int i = 0; i <= j; ++i) {
                const double* vi = linalg::col(V, n, i);
                double h = linalg::dot(n, w.data(), vi);
                H[j * (restart + 1) + i] = h;
                linalg::axpy(n, -h, vi, w.data());
            }
            double h_next = linalg::nrm2(w);
            H[j * (restart + 1) + j + 1] = h_next;

            // v_{j+1} = w / h_next
            if (h_next > 0.0) {
                double* vj1 = linalg::col(V, n, j + 1);
                linalg::copy(n, w.data(), vj1);
                linalg::scal(n, 1.0 / h_next, vj1);
            }

            // Apply previous Givens rotations to column j of H
            for (int i = 0; i < j; ++i) {
                double h_i   = H[j * (restart + 1) + i];
                double h_ip1 = H[j * (restart + 1) + i + 1];
                H[j * (restart + 1) + i]     =  cs[i] * h_i + sn[i] * h_ip1;
                H[j * (restart + 1) + i + 1] = -sn[i] * h_i + cs[i] * h_ip1;
            }

            // Compute new Givens rotation to zero H[j+1, j]
            double h_jj   = H[j * (restart + 1) + j];
            double h_jp1j = H[j * (restart + 1) + j + 1];
            double denom  = std::hypot(h_jj, h_jp1j);
            if (denom < 1e-300) { cs[j] = 1.0; sn[j] = 0.0; }
            else {
                cs[j] = h_jj   / denom;
                sn[j] = h_jp1j / denom;
            }
            H[j * (restart + 1) + j]     = cs[j] * h_jj + sn[j] * h_jp1j;
            H[j * (restart + 1) + j + 1] = 0.0;

            // Update RHS
            g[j + 1] = -sn[j] * g[j];
            g[j]     =  cs[j] * g[j];

            rel_res = std::abs(g[j + 1]) / r0_norm;
            if (rel_res < tol) { j++; converged = true; break; }
        }

        // Back-substitute: solve upper triangular H[0:j, 0:j] * y = g[0:j]
        int m = j; // number of steps completed
        Vec y(m);
        for (int i = m - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int k = i + 1; k < m; ++k)
                y[i] -= H[k * (restart + 1) + i] * y[k];
            y[i] /= H[i * (restart + 1) + i];
        }

        // x <- x + V[:,0:m] * y
        for (int i = 0; i < m; ++i)
            linalg::axpy(n, y[i], linalg::col(V, n, i), x.data());

        if (converged) break;

        // Recompute residual for next cycle
        matvec(x, Ax);
        ++total_mv;
        linalg::copy(b, r);
        linalg::axpy(-1.0, Ax, r);
    }

    // Final relative residual (recomputed accurately)
    matvec(x, Ax);
    linalg::copy(b, r);
    linalg::axpy(-1.0, Ax, r);
    rel_res = linalg::nrm2(r) / r0_norm;

    return {total_mv, rel_res, rel_res < tol};
}

} // namespace gmres
