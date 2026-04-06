#pragma once
// linalg.h — vector/matrix operations abstraction
// Backend: STL by default, BLAS when USE_BLAS is defined.

#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstring>

#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif

namespace linalg {

using Vec = std::vector<double>;
using Mat = std::vector<double>; // column-major, shape [rows x cols]

// ---------------------------------------------------------------------------
// dot : returns x^T y
// ---------------------------------------------------------------------------
inline double dot(int n, const double* x, const double* y) {
#ifdef USE_BLAS
    return cblas_ddot(n, x, 1, y, 1);
#else
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
#endif
}
inline double dot(const Vec& x, const Vec& y) {
    return dot(static_cast<int>(x.size()), x.data(), y.data());
}

// ---------------------------------------------------------------------------
// nrm2 : returns ||x||_2
// ---------------------------------------------------------------------------
inline double nrm2(int n, const double* x) {
#ifdef USE_BLAS
    return cblas_dnrm2(n, x, 1);
#else
    return std::sqrt(dot(n, x, x));
#endif
}
inline double nrm2(const Vec& x) {
    return nrm2(static_cast<int>(x.size()), x.data());
}

// ---------------------------------------------------------------------------
// axpy : y <- alpha*x + y
// ---------------------------------------------------------------------------
inline void axpy(int n, double alpha, const double* x, double* y) {
#ifdef USE_BLAS
    cblas_daxpy(n, alpha, x, 1, y, 1);
#else
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
#endif
}
inline void axpy(double alpha, const Vec& x, Vec& y) {
    axpy(static_cast<int>(x.size()), alpha, x.data(), y.data());
}

// ---------------------------------------------------------------------------
// scal : x <- alpha*x
// ---------------------------------------------------------------------------
inline void scal(int n, double alpha, double* x) {
#ifdef USE_BLAS
    cblas_dscal(n, alpha, x, 1);
#else
    for (int i = 0; i < n; ++i) x[i] *= alpha;
#endif
}
inline void scal(double alpha, Vec& x) {
    scal(static_cast<int>(x.size()), alpha, x.data());
}

// ---------------------------------------------------------------------------
// copy : dst <- src
// ---------------------------------------------------------------------------
inline void copy(int n, const double* src, double* dst) {
#ifdef USE_BLAS
    cblas_dcopy(n, src, 1, dst, 1);
#else
    std::memcpy(dst, src, n * sizeof(double));
#endif
}
inline void copy(const Vec& src, Vec& dst) {
    dst.resize(src.size());
    copy(static_cast<int>(src.size()), src.data(), dst.data());
}

// ---------------------------------------------------------------------------
// gemv : y <- alpha*A*x + beta*y
// A is stored column-major, shape [m x n]
// ---------------------------------------------------------------------------
inline void gemv(int m, int n,
                 double alpha, const double* A,
                 const double* x,
                 double beta,  double* y) {
#ifdef USE_BLAS
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                m, n, alpha, A, m, x, 1, beta, y, 1);
#else
    // y <- beta*y first
    for (int i = 0; i < m; ++i) y[i] *= beta;
    // y <- y + alpha * A * x
    for (int j = 0; j < n; ++j) {
        double axj = alpha * x[j];
        for (int i = 0; i < m; ++i)
            y[i] += axj * A[j * m + i];
    }
#endif
}
inline void gemv(int m, int n,
                 double alpha, const Mat& A,
                 const Vec& x,
                 double beta,  Vec& y) {
    y.resize(m);
    gemv(m, n, alpha, A.data(), x.data(), beta, y.data());
}

// ---------------------------------------------------------------------------
// gemm : C <- alpha*A*B + beta*C
// All column-major. A:[m x k], B:[k x n], C:[m x n]
// ---------------------------------------------------------------------------
inline void gemm(int m, int k, int n,
                 double alpha, const double* A,
                               const double* B,
                 double beta,        double* C) {
#ifdef USE_BLAS
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, m, B, k, beta, C, m);
#else
    // C <- beta*C
    int sz = m * n;
    for (int i = 0; i < sz; ++i) C[i] *= beta;
    // C <- C + alpha * A * B
    for (int j = 0; j < n; ++j)
        for (int p = 0; p < k; ++p) {
            double ab = alpha * B[j * k + p];
            for (int i = 0; i < m; ++i)
                C[j * m + i] += ab * A[p * m + i];
        }
#endif
}
inline void gemm(int m, int k, int n,
                 double alpha, const Mat& A, const Mat& B,
                 double beta,        Mat& C) {
    C.resize(m * n);
    gemm(m, k, n, alpha, A.data(), B.data(), beta, C.data());
}

// ---------------------------------------------------------------------------
// Helpers: zero-init
// ---------------------------------------------------------------------------
inline Vec make_vec(int n, double val = 0.0) { return Vec(n, val); }
inline Mat make_mat(int rows, int cols, double val = 0.0) {
    return Mat(rows * cols, val);
}

// Column accessor (column-major matrix stored as flat vector)
inline double* col(Mat& M, int rows, int j) { return M.data() + j * rows; }
inline const double* col(const Mat& M, int rows, int j) { return M.data() + j * rows; }

} // namespace linalg
