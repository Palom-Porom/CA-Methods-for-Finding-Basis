#pragma once
// ilu.h — Incomplete LU factorization with zero fill-in (ILU(0))
//
// Build:  ILU0 f = build_ilu0(A);
// Apply:  apply_ilu0(f, r, z);   // solves L*U*z = r
//
// The sparsity pattern of L+U is identical to that of A.
// L has an implicit unit diagonal (stored values are the sub-diagonal part).
// U stores the diagonal and super-diagonal part.
//
// Intended use: right-preconditioned GMRES
//   solve  A * (M^{-1} u) = b,  then x = M^{-1} u
//   where  M = L * U

#include "matrix_market.h"   // CSRMatrix, Vec
#include "linalg.h"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace ilu {

using Vec = linalg::Vec;

// ---------------------------------------------------------------------------
// ILU(0) factors stored in a single CSR structure mirroring A's pattern.
// Convention (same as ILUT / Saad "Iterative Methods"):
//   values[k] for col_idx[k] <  row  → L factor (L has implicit unit diag)
//   values[k] for col_idx[k] >= row  → U factor (diagonal included in U)
// ---------------------------------------------------------------------------
struct ILU0 {
    int                 n;
    std::vector<double> values;   // same sparsity pattern as A
    std::vector<int>    col_idx;  // copied from A
    std::vector<int>    row_ptr;  // copied from A
    std::vector<int>    diag_idx; // row_ptr-based index of diagonal entry per row
};

// ---------------------------------------------------------------------------
// build_ilu0
//
// Performs in-place IKJ variant of ILU(0).
// Throws if A is not square or a zero pivot is encountered.
// ---------------------------------------------------------------------------
inline ILU0 build_ilu0(const CSRMatrix& A)
{
    if (A.rows != A.cols)
        throw std::runtime_error("build_ilu0: matrix must be square");

    const int n = A.rows;

    ILU0 f;
    f.n       = n;
    f.values  = A.values;    // working copy — will be overwritten
    f.col_idx = A.col_idx;
    f.row_ptr = A.row_ptr;
    f.diag_idx.resize(n, -1);

    // IKJ ILU(0) requires col_idx sorted within every row.
    // Some .mtx files (especially after symmetric expansion) are not sorted.
    for (int i = 0; i < n; ++i) {
        const int start = f.row_ptr[i], end = f.row_ptr[i + 1];
        bool sorted = true;
        for (int k = start + 1; k < end; ++k)
            if (f.col_idx[k] < f.col_idx[k - 1]) { sorted = false; break; }
        if (!sorted) {
            std::vector<int> idx(end - start);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b){
                          return f.col_idx[start + a] < f.col_idx[start + b];
                      });
            std::vector<double> tv(end - start);
            std::vector<int>    tc(end - start);
            for (int k = 0; k < (int)idx.size(); ++k) {
                tv[k] = f.values[start + idx[k]];
                tc[k] = f.col_idx[start + idx[k]];
            }
            for (int k = start; k < end; ++k) {
                f.values[k]  = tv[k - start];
                f.col_idx[k] = tc[k - start];
            }
        }
    }

    // Locate diagonal position for every row
    for (int i = 0; i < n; ++i) {
        for (int k = f.row_ptr[i]; k < f.row_ptr[i + 1]; ++k) {
            if (f.col_idx[k] == i) { f.diag_idx[i] = k; break; }
        }
        if (f.diag_idx[i] == -1)
            throw std::runtime_error(
                "build_ilu0: missing diagonal at row " + std::to_string(i));
    }

    // IKJ ILU(0) — for each row i, eliminate columns k < i
    for (int i = 1; i < n; ++i) {
        for (int ptr = f.row_ptr[i]; ptr < f.diag_idx[i]; ++ptr) {
            const int k = f.col_idx[ptr];   // column index (k < i)

            // multiplier: a_ik /= a_kk
            const double ukk = f.values[f.diag_idx[k]];
            if (std::abs(ukk) < 1e-300)
                throw std::runtime_error(
                    "build_ilu0: zero pivot at row " + std::to_string(k));
            f.values[ptr] /= ukk;
            const double m = f.values[ptr];

            // update row i using row k — only positions present in pattern of A
            // walk row k from diag+1 (U part) and row i simultaneously
            int pi = f.diag_idx[i];           // start of U part of row i
            int pk = f.diag_idx[k] + 1;       // start of U part of row k

            while (pk < f.row_ptr[k + 1] && pi < f.row_ptr[i + 1]) {
                const int jk = f.col_idx[pk];
                const int ji = f.col_idx[pi];
                if (jk == ji) {
                    f.values[pi] -= m * f.values[pk];
                    ++pk; ++pi;
                } else if (jk < ji) {
                    ++pk;   // column not in pattern of row i → skip (zero fill)
                } else {
                    ++pi;
                }
            }
        }
    }

    return f;
}

// ---------------------------------------------------------------------------
// apply_ilu0
//
// Solves  L * U * z = r  via:
//   1. Forward substitution:   L * w = r   (L has unit diagonal)
//   2. Backward substitution:  U * z = w
// ---------------------------------------------------------------------------
inline void apply_ilu0(const ILU0& f, const Vec& r, Vec& z)
{
    const int n = f.n;
    z.resize(n);

    // --- Forward: L * w = r  (store w in z) ---
    for (int i = 0; i < n; ++i) {
        double s = r[i];
        for (int k = f.row_ptr[i]; k < f.diag_idx[i]; ++k)
            s -= f.values[k] * z[f.col_idx[k]];
        z[i] = s;   // L diagonal is 1
    }

    // --- Backward: U * z = w ---
    for (int i = n - 1; i >= 0; --i) {
        double s = z[i];
        for (int k = f.diag_idx[i] + 1; k < f.row_ptr[i + 1]; ++k)
            s -= f.values[k] * z[f.col_idx[k]];
        z[i] = s / f.values[f.diag_idx[i]];
    }
}

// ---------------------------------------------------------------------------
// make_precond — wraps ILU0 into the Precond functor expected by gmres::solve
//
// Usage:
//   ILU0 f = build_ilu0(csr);
//   auto P = ilu::make_precond(f);
//   gmres::solve(mv, b, x, restart, tol, maxiter, P);
//
// IMPORTANT: f must outlive P.
// ---------------------------------------------------------------------------
inline std::function<void(const Vec&, Vec&)> make_precond(const ILU0& f)
{
    return [&f](const Vec& r, Vec& z) { apply_ilu0(f, r, z); };
}

// ===========================================================================
// Diagonal scaling + ILU(0)  (sILU0)
// ===========================================================================

// ---------------------------------------------------------------------------
// scale_csr
//
// Symmetrically scales A by the inverse square root of its diagonal:
//   d[i]             = 1 / sqrt(A[i,i])
//   A_scaled[i,j]    = d[i] * A[i,j] * d[j]
//
// Returns the scaled matrix and the vector d.
// Throws if any diagonal entry is non-positive.
// ---------------------------------------------------------------------------
inline std::pair<CSRMatrix, Vec> scale_csr(const CSRMatrix& A)
{
    const int n = A.rows;

    // Build d[i] = 1 / sqrt(A[i,i])
    Vec d(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double diag_val = 0.0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            if (A.col_idx[k] == i) { diag_val = A.values[k]; break; }
        }
        if (diag_val <= 0.0)
            throw std::runtime_error(
                "scale_csr: non-positive diagonal at row " + std::to_string(i));
        d[i] = 1.0 / std::sqrt(diag_val);
    }

    // Build scaled matrix: A_scaled[i,j] = d[i] * A[i,j] * d[j]
    CSRMatrix As;
    As.rows    = A.rows;
    As.cols    = A.cols;
    As.nnz     = A.nnz;
    As.row_ptr = A.row_ptr;
    As.col_idx = A.col_idx;
    As.values  = A.values;
    for (int i = 0; i < n; ++i) {
        for (int k = As.row_ptr[i]; k < As.row_ptr[i + 1]; ++k)
            As.values[k] *= d[i] * d[As.col_idx[k]];
    }

    return {std::move(As), std::move(d)};
}

// ---------------------------------------------------------------------------
// ScaledILU0 — ILU(0) of the diagonally-scaled matrix together with d.
// ---------------------------------------------------------------------------
struct ScaledILU0 {
    ILU0 f;   // ILU(0) factors of D*A*D
    Vec  d;   // d[i] = 1/sqrt(A[i,i])
};

// ---------------------------------------------------------------------------
// build_milu0
//
// Modified ILU(0): like ILU(0) but dropped fill-in entries are compensated
// by adding their sum to the diagonal (preserves row sums of A).
//
// For SPD and H-matrices this guarantees positive pivots and a much better
// preconditioner than plain ILU(0) on matrices that are not diagonally
// dominant.
// ---------------------------------------------------------------------------
inline ILU0 build_milu0(const CSRMatrix& A)
{
    if (A.rows != A.cols)
        throw std::runtime_error("build_milu0: matrix must be square");

    const int n = A.rows;
    ILU0 f;
    f.n       = n;
    f.values  = A.values;
    f.col_idx = A.col_idx;
    f.row_ptr = A.row_ptr;
    f.diag_idx.resize(n, -1);

    // Sort column indices within each row (same as build_ilu0)
    for (int i = 0; i < n; ++i) {
        const int start = f.row_ptr[i], end = f.row_ptr[i + 1];
        bool sorted = true;
        for (int k = start + 1; k < end; ++k)
            if (f.col_idx[k] < f.col_idx[k - 1]) { sorted = false; break; }
        if (!sorted) {
            std::vector<int> idx(end - start);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b){ return f.col_idx[start+a] < f.col_idx[start+b]; });
            std::vector<double> tv(end - start);
            std::vector<int>    tc(end - start);
            for (int k = 0; k < (int)idx.size(); ++k) {
                tv[k] = f.values[start + idx[k]];
                tc[k] = f.col_idx[start + idx[k]];
            }
            for (int k = start; k < end; ++k) { f.values[k] = tv[k-start]; f.col_idx[k] = tc[k-start]; }
        }
    }

    // Locate diagonal entries
    for (int i = 0; i < n; ++i) {
        for (int k = f.row_ptr[i]; k < f.row_ptr[i + 1]; ++k)
            if (f.col_idx[k] == i) { f.diag_idx[i] = k; break; }
        if (f.diag_idx[i] == -1)
            throw std::runtime_error(
                "build_milu0: missing diagonal at row " + std::to_string(i));
    }

    // IKJ MILU(0): same as ILU(0) but dropped fill is added back to diagonal
    for (int i = 1; i < n; ++i) {
        for (int ptr = f.row_ptr[i]; ptr < f.diag_idx[i]; ++ptr) {
            const int k = f.col_idx[ptr];
            const double ukk = f.values[f.diag_idx[k]];
            if (std::abs(ukk) < 1e-300)
                throw std::runtime_error(
                    "build_milu0: zero pivot at row " + std::to_string(k));
            f.values[ptr] /= ukk;
            const double m = f.values[ptr];

            int pi = f.diag_idx[i];
            int pk = f.diag_idx[k] + 1;
            double drop = 0.0;   // accumulated dropped fill-in

            while (pk < f.row_ptr[k + 1] && pi < f.row_ptr[i + 1]) {
                const int jk = f.col_idx[pk];
                const int ji = f.col_idx[pi];
                if (jk == ji) {
                    f.values[pi] -= m * f.values[pk];
                    ++pk; ++pi;
                } else if (jk < ji) {
                    drop += m * f.values[pk];   // dropped → goes to diagonal
                    ++pk;
                } else {
                    ++pi;
                }
            }
            // Any remaining entries in row k have no matching column in row i
            while (pk < f.row_ptr[k + 1]) {
                drop += m * f.values[pk];
                ++pk;
            }
            // Compensate diagonal with dropped fill-in (MILU correction)
            f.values[f.diag_idx[i]] -= drop;
        }
    }

    return f;
}

// ---------------------------------------------------------------------------
// build_scaled_ilu0
//
// Scales A symmetrically (D*A*D, d[i]=1/sqrt(A[i,i])), then builds
// MILU(0) on the scaled matrix.
//
// For matrices that are not diagonally dominant after scaling (common for
// ill-conditioned SPD problems), MILU(0) can still produce negative U
// pivots.  To handle this we search for the minimum diagonal shift s ≥ 0
// such that MILU(0) of (A_scaled + s·I) has all positive pivots, then
// apply that shift before the final factorization.
//
// Algorithm:
//   1. Try MILU(0) with no shift; if all pivots > 0, done.
//   2. Otherwise compute the Gershgorin excess G = max_i(Σ_{j≠i}|a_ij|-1)
//      as an upper bound, then binary-search in [0, G] for the minimum
//      positive-pivot shift (30 steps ≈ G·2^{-30} precision).
//   3. Apply the found shift with a 5 % safety margin.
// ---------------------------------------------------------------------------
inline ScaledILU0 build_scaled_ilu0(const CSRMatrix& A)
{
    auto [As, d] = scale_csr(A);
    const int n = As.rows;

    // Returns true iff MILU(0) of (As + shift*I) has all positive pivots.
    // Makes a temporary copy of As to avoid modifying the working matrix.
    auto all_positive = [&](double shift) -> bool {
        CSRMatrix Ash = As;   // cheap copy of scaled matrix
        if (shift > 0.0)
            for (int i = 0; i < n; ++i)
                for (int k = Ash.row_ptr[i]; k < Ash.row_ptr[i + 1]; ++k)
                    if (Ash.col_idx[k] == i) { Ash.values[k] += shift; break; }
        try {
            ILU0 f = build_milu0(Ash);
            for (int i = 0; i < n; ++i)
                if (f.values[f.diag_idx[i]] <= 0.0) return false;
            return true;
        } catch (...) { return false; }
    };

    double shift = 0.0;

    if (!all_positive(0.0)) {
        // Upper bound: Gershgorin excess (guarantees diagonal dominance)
        double hi = 0.0;
        for (int i = 0; i < n; ++i) {
            double off = 0.0;
            for (int k = As.row_ptr[i]; k < As.row_ptr[i + 1]; ++k)
                if (As.col_idx[k] != i) off += std::abs(As.values[k]);
            hi = std::max(hi, off - 1.0);   // scaled diagonal is 1
        }
        hi = std::max(hi, 1e-10);

        // Binary search: 30 steps give ~2^{-30} relative precision
        double lo = 0.0;
        for (int iter = 0; iter < 30; ++iter) {
            double mid = 0.5 * (lo + hi);
            if (all_positive(mid)) hi = mid;
            else                   lo = mid;
        }
        shift = hi * 1.05;   // 5 % safety margin above minimum
    }

    // Apply the determined shift to the working scaled matrix
    if (shift > 0.0)
        for (int i = 0; i < n; ++i)
            for (int k = As.row_ptr[i]; k < As.row_ptr[i + 1]; ++k)
                if (As.col_idx[k] == i) { As.values[k] += shift; break; }

    ScaledILU0 sf;
    sf.f = build_milu0(As);
    sf.d = std::move(d);
    return sf;
}

// ---------------------------------------------------------------------------
// apply_scaled_ilu0
//
// Applies the scaled preconditioner M^{-1} to r:
//   tmp  = D * r          (tmp[i]  = d[i] * r[i])
//   w    = (LU)^{-1} * tmp
//   z    = D * w          (z[i]    = d[i] * w[i])
//
// This is the correct right-preconditioner for A derived from ILU(0) of D*A*D:
//   A * (D*(LU)^{-1}*D) ≈ I
// ---------------------------------------------------------------------------
inline void apply_scaled_ilu0(const ScaledILU0& sf, const Vec& r, Vec& z)
{
    const int n = sf.f.n;
    Vec tmp(n);
    for (int i = 0; i < n; ++i) tmp[i] = sf.d[i] * r[i];
    apply_ilu0(sf.f, tmp, z);
    for (int i = 0; i < n; ++i) z[i] *= sf.d[i];
}

// ---------------------------------------------------------------------------
// make_scaled_precond — wraps ScaledILU0 into a Precond functor.
//
// IMPORTANT: sf must outlive the returned functor.
// ---------------------------------------------------------------------------
inline std::function<void(const Vec&, Vec&)> make_scaled_precond(const ScaledILU0& sf)
{
    return [&sf](const Vec& r, Vec& z) { apply_scaled_ilu0(sf, r, z); };
}

} // namespace ilu