#pragma once
// ca_gmres.h — Communication-Avoiding GMRES (s-step CA-GMRES)

#include "linalg.h"
#include <functional>
#include <cmath>
#include <vector>
#include <cassert>

namespace ca_gmres {

using Vec    = linalg::Vec;
using Mat    = linalg::Mat;
using Matvec = std::function<void(const Vec& x, Vec& y)>;

// Вспомогательная функция: Блочный Грам-Шмидт с переортогонализацией (CGS2)
inline void block_orthogonalise(
    const Mat& Q, int n, int q_cols,
    Mat& K, int k_cols,
    Mat& B, Mat& R)
    {
        B.assign(q_cols * k_cols, 0.0);
        Mat B_temp(q_cols * k_cols, 0.0);

        auto compute_QtK = [&](Mat& out_B) {
    #ifdef USE_BLAS
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        q_cols, k_cols, n, 1.0, Q.data(), n, K.data(), n, 0.0, out_B.data(), q_cols);
    #else
            for (int j = 0; j < k_cols; ++j) {
                for (int i = 0; i < q_cols; ++i) {
                    double s = 0.0;
                    for (int r = 0; r < n; ++r) s += Q[i * n + r] * K[j * n + r];
                    out_B[j * q_cols + i] = s;
                }
            }
    #endif
        };

        // Проход 1
        compute_QtK(B_temp);
        linalg::gemm(n, q_cols, k_cols, -1.0, Q, B_temp, 1.0, K);
        for (int i = 0; i < q_cols * k_cols; ++i) B[i] += B_temp[i];

        // Проход 2 (Переортогонализация для борьбы с потерей точности)
        compute_QtK(B_temp);
        linalg::gemm(n, q_cols, k_cols, -1.0, Q, B_temp, 1.0, K);
        for (int i = 0; i < q_cols * k_cols; ++i) B[i] += B_temp[i];

        // Внутренний тонкий QR для блока K (тоже с переортогонализацией)
        R.assign(k_cols * k_cols, 0.0);
        for (int j = 0; j < k_cols; ++j) {
            double* kj = linalg::col(K, n, j);
            for (int iter = 0; iter < 2; ++iter) {
                for (int i = 0; i < j; ++i) {
                    const double* ki = linalg::col(K, n, i);
                    double h = linalg::dot(n, kj, ki);
                    if (iter == 0) R[j * k_cols + i] = h;
                    else           R[j * k_cols + i] += h;
                    linalg::axpy(n, -h, ki, kj);
                }
            }
            double nrm = linalg::nrm2(n, kj);
            R[j * k_cols + j] = nrm;
            if (nrm > 1e-300) linalg::scal(n, 1.0 / nrm, kj);
        }
    }

    struct Result {
        int    iters;
        double rel_res;
        bool   converged;
    };

    inline Result solve(
        const Matvec& matvec, const Vec& b, Vec& x,
        int s = 4, int restart = 30, double tol = 1e-10, int max_iter = 1000)
    {
        const int n = static_cast<int>(b.size());
        assert(static_cast<int>(x.size()) == n);

        int m = (restart / s) * s;
        if (m == 0) m = s;

        Vec Ax(n), r(n);
        matvec(x, Ax);
        linalg::copy(b, r);
        linalg::axpy(-1.0, Ax, r);

        double r0_norm = linalg::nrm2(r);
        if (r0_norm == 0.0) return {0, 0.0, true};

        int total_mv = 1;
        double rel_res = 1.0;
        bool converged = false;

        Mat Q = linalg::make_mat(n, m + 1);
        Mat Hmat = linalg::make_mat(m + 1, m, 0.0); // Чистая матрица Арнольди
        Mat Rmat = linalg::make_mat(m + 1, m, 0.0); // Треугольная после Гивенса
        
        std::vector<double> cs(m), sn(m);
        Vec g(m + 1);

        while (total_mv < max_iter && !converged) {
            double beta = linalg::nrm2(r);
            rel_res = beta / r0_norm;
            if (rel_res < tol) { converged = true; break; }

            double* q0 = linalg::col(Q, n, 0);
            linalg::copy(n, r.data(), q0);
            linalg::scal(n, 1.0 / beta, q0);

            std::fill(g.begin(), g.end(), 0.0);
            g[0] = beta;
            std::fill(Hmat.begin(), Hmat.end(), 0.0);
            std::fill(Rmat.begin(), Rmat.end(), 0.0);

            int q_cols = 1;
            int j_total = 0;
            bool inner_conv = false;

            for (int block = 0; block * s < m && !inner_conv && total_mv < max_iter; ++block) {
                int j0 = block * s;
                int s_actual = std::min(s, m - j0);

                Mat K = linalg::make_mat(n, s_actual + 1);
                Vec v_start(linalg::col(Q, n, q_cols - 1), linalg::col(Q, n, q_cols - 1) + n);
                linalg::copy(n, v_start.data(), linalg::col(K, n, 0));
                
                // Динамический сдвиг: оцениваем спектр, чтобы базис не вырождался
                double shift = 0.0;
                if (s_actual > 0) {
                    Vec vout(n);
                    matvec(v_start, vout);
                    shift = linalg::dot(n, v_start.data(), vout.data()); // Коэффициент Рэлея
                    
                    linalg::axpy(n, -shift, v_start.data(), vout.data());
                    linalg::copy(n, vout.data(), linalg::col(K, n, 1));
                    ++total_mv;
                }
                // Формируем базис: v_{p+1} = (A - shift * I) v_p
                for (int p = 1; p < s_actual; ++p) {
                    Vec vin(linalg::col(K, n, p), linalg::col(K, n, p) + n);
                    Vec vout(n);
                    matvec(vin, vout);
                    linalg::axpy(n, -shift, vin.data(), vout.data());
                    linalg::copy(n, vout.data(), linalg::col(K, n, p + 1));
                    ++total_mv;
                }

                Mat K_new = linalg::make_mat(n, s_actual);
                for (int p = 0; p < s_actual; ++p)
                    linalg::copy(n, linalg::col(K, n, p + 1), linalg::col(K_new, n, p));

                Mat Q_cur = linalg::make_mat(n, q_cols);
                for (int p = 0; p < q_cols; ++p)
                    linalg::copy(n, linalg::col(Q, n, p), linalg::col(Q_cur, n, p));

                Mat B_block, R_block;
                block_orthogonalise(Q_cur, n, q_cols, K_new, s_actual, B_block, R_block);

                // Собираем матрицу перехода U
                Mat U = linalg::make_mat(m + 1, s_actual + 1, 0.0);
                U[0 * (m + 1) + j0] = 1.0; 

                for (int p = 0; p < s_actual; ++p) {
                    int col_u = p + 1;
                    for (int i = 0; i < q_cols; ++i) U[col_u * (m + 1) + i] = B_block[p * q_cols + i];
                    for (int i = 0; i <= p; ++i)     U[col_u * (m + 1) + q_cols + i] = R_block[p * s_actual + i];
                }

                // Математически строгое восстановление матрицы Хессенберга со сдвигом
                for (int p = 0; p < s_actual; ++p) {
                    int k = j0 + p;
                    double denom = U[p * (m + 1) + k];
                    
                    if (std::abs(denom) < 1e-300) {
                        for (int r = 0; r <= k + 1; ++r) Hmat[k * (m + 1) + r] = 0.0;
                    } else {
                        for (int r = 0; r <= k + 1; ++r) {
                            double sum = 0.0;
                            for (int c = 0; c < k; ++c) {
                                sum += Hmat[c * (m + 1) + r] * U[p * (m + 1) + c];
                            }
                            
                            double u_p   = U[p * (m + 1) + r];
                            double u_pp1 = U[(p + 1) * (m + 1) + r];
                            
                            // H * u_p = u_{p+1} + shift * u_p
                            Hmat[k * (m + 1) + r] = (u_pp1 + shift * u_p - sum) / denom;
                        }
                    }

                    // Копируем найденный столбец в Rmat для последующего вращения
                    for (int r = 0; r <= k + 1; ++r) {
                        Rmat[k * (m + 1) + r] = Hmat[k * (m + 1) + r];
                    }
                }

                for (int p = 0; p < s_actual; ++p)
                    linalg::copy(n, linalg::col(K_new, n, p), linalg::col(Q, n, q_cols + p));

                // Применяем вращения Гивенса к Rmat
                for (int p = 0; p < s_actual && !inner_conv; ++p) {
                    int jj = j0 + p;

                    for (int i = 0; i < jj; ++i) {
                        double hi   = Rmat[jj * (m + 1) + i];
                        double hip1 = Rmat[jj * (m + 1) + i + 1];
                        Rmat[jj * (m + 1) + i]     =  cs[i] * hi + sn[i] * hip1;
                        Rmat[jj * (m + 1) + i + 1] = -sn[i] * hi + cs[i] * hip1;
                    }

                    double hjj   = Rmat[jj * (m + 1) + jj];
                    double hjp1j = Rmat[jj * (m + 1) + jj + 1];
                    double rot_denom = std::hypot(hjj, hjp1j);
                    if (rot_denom < 1e-300) { cs[jj] = 1.0; sn[jj] = 0.0; }
                    else { cs[jj] = hjj / rot_denom; sn[jj] = hjp1j / rot_denom; }
                    
                    Rmat[jj * (m + 1) + jj]     = cs[jj] * hjj + sn[jj] * hjp1j;
                    Rmat[jj * (m + 1) + jj + 1] = 0.0;

                    g[jj + 1] = -sn[jj] * g[jj];
                    g[jj]     =  cs[jj] * g[jj];

                    rel_res = std::abs(g[jj + 1]) / r0_norm;
                    if (rel_res < tol) {
                        j_total = jj + 1;
                        inner_conv = true;
                        converged  = true;
                    }
                }

                q_cols += s_actual;
                if (!inner_conv) j_total = j0 + s_actual;
            }

            int mm = j_total;
            Vec y(mm);
            for (int i = mm - 1; i >= 0; --i) {
                y[i] = g[i];
                for (int k = i + 1; k < mm; ++k)
                    y[i] -= Rmat[k * (m + 1) + i] * y[k];
                if (std::abs(Rmat[i * (m + 1) + i]) > 1e-300)
                    y[i] /= Rmat[i * (m + 1) + i];
            }

            for (int i = 0; i < mm; ++i)
                linalg::axpy(n, y[i], linalg::col(Q, n, i), x.data());

            if (converged) break;

            matvec(x, Ax);
            ++total_mv;
            linalg::copy(b, r);
            linalg::axpy(-1.0, Ax, r);
        }

        matvec(x, Ax);
        linalg::copy(b, r);
        linalg::axpy(-1.0, Ax, r);
        rel_res = linalg::nrm2(r) / r0_norm;

        return {total_mv, rel_res, rel_res < tol};
    }

} // namespace ca_gmres