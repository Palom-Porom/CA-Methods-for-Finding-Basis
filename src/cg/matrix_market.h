#pragma once
// matrix_market.h — Matrix Market (.mtx) parser
// Produces a CSRMatrix usable directly with gmres::solve() via spmv_matvec()

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <functional>

// ---------------------------------------------------------------------------
// CSR matrix
// ---------------------------------------------------------------------------
struct CSRMatrix {
    int rows = 0;
    int cols = 0;
    int nnz  = 0;
    std::vector<double> values;   // length nnz
    std::vector<int>    col_idx;  // length nnz
    std::vector<int>    row_ptr;  // length rows+1
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace mm_detail {

struct MMHeader {
    bool is_coordinate = false;
    bool is_real       = false;
    bool is_integer    = false;
    bool is_pattern    = false;
    bool is_complex    = false;
    bool is_general    = false;
    bool is_symmetric  = false;
    bool is_skew       = false;
    bool is_hermitian  = false;
};

inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

inline MMHeader parse_banner(const std::string& line) {
    std::istringstream ss(line);
    std::string token;
    std::vector<std::string> tokens;
    while (ss >> token) tokens.push_back(to_lower(token));

    if (tokens.size() < 5)
        throw std::runtime_error("MM banner: expected 5 tokens, got " +
                                 std::to_string(tokens.size()));
    if (tokens[0] != "%%matrixmarket")
        throw std::runtime_error("MM banner: must start with %%MatrixMarket");
    if (tokens[1] != "matrix")
        throw std::runtime_error("MM: only 'matrix' object supported");

    MMHeader h;
    h.is_coordinate = (tokens[2] == "coordinate");
    h.is_real       = (tokens[3] == "real");
    h.is_integer    = (tokens[3] == "integer");
    h.is_pattern    = (tokens[3] == "pattern");
    h.is_complex    = (tokens[3] == "complex");
    h.is_general    = (tokens[4] == "general");
    h.is_symmetric  = (tokens[4] == "symmetric");
    h.is_skew       = (tokens[4] == "skew-symmetric");
    h.is_hermitian  = (tokens[4] == "hermitian");

    if (!h.is_coordinate)
        throw std::runtime_error("MM: only 'coordinate' (sparse) format supported");
    if (h.is_complex)
        throw std::runtime_error("MM: complex matrices not supported");

    return h;
}

} // namespace mm_detail

// ---------------------------------------------------------------------------
// load_matrix_market
// ---------------------------------------------------------------------------
inline CSRMatrix load_matrix_market(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    std::string line;

    // 1. Banner
    if (!std::getline(f, line))
        throw std::runtime_error("MM: empty file");
    auto header = mm_detail::parse_banner(line);

    // 2. Skip comment lines
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] != '%') break;
    }

    // 3. Size line  — line is already the first non-comment line
    int M, N, L;
    {
        std::istringstream ss(line);
        if (!(ss >> M >> N >> L))
            throw std::runtime_error("MM: cannot parse size line: " + line);
    }
    if (M <= 0 || N <= 0 || L <= 0)
        throw std::runtime_error("MM: invalid dimensions");

    // 4. Read triplets
    struct Triplet { int r, c; double v; };
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<size_t>(L) * (header.is_general ? 1 : 2));

    for (int k = 0; k < L; ++k) {
        // skip blank / stray comment lines (non-standard but seen in the wild)
        while (true) {
            if (!std::getline(f, line))
                throw std::runtime_error(
                    "MM: unexpected end of file at entry " + std::to_string(k));
            if (!line.empty() && line[0] != '%') break;
        }

        std::istringstream ss(line);
        int r, c;
        double v = 1.0;  // default for pattern matrices

        // BUG FIX (was in previous version):
        // operator>> on int stops at the decimal point, so "1.0 2.0 3.5"
        // would parse r=1 correctly but leave ".0" in the stream for c.
        // Some exporters write integer indices with a trailing ".0".
        // Solution: read as double first, then cast.
        double rd, cd;
        if (!(ss >> rd >> cd))
            throw std::runtime_error("MM: cannot parse indices: " + line);
        r = static_cast<int>(rd) - 1;  // convert to 0-based
        c = static_cast<int>(cd) - 1;

        if (!header.is_pattern && !(ss >> v))
            throw std::runtime_error("MM: cannot parse value: " + line);

        if (r < 0 || r >= M || c < 0 || c >= N)
            throw std::runtime_error(
                "MM: index out of range at entry " + std::to_string(k) +
                " (" + std::to_string(r+1) + "," + std::to_string(c+1) + ")");

        triplets.push_back({r, c, v});

        // Expand symmetry (skip diagonal)
        if (!header.is_general && r != c) {
            double v2 = header.is_skew ? -v : v;
            triplets.push_back({c, r, v2});
        }
    }

    // 5. Sort by (row, col)
    std::sort(triplets.begin(), triplets.end(),
              [](const Triplet& a, const Triplet& b){
                  return a.r != b.r ? a.r < b.r : a.c < b.c;
              });

    // 6. Build CSR, merging duplicate (r,c) by summing
    CSRMatrix mat;
    mat.rows = M;
    mat.cols = N;
    mat.row_ptr.resize(M + 1, 0);

    // Pass 1: count unique (r,c) pairs per row
    {
        int prev_r = -1, prev_c = -1;
        for (auto& t : triplets) {
            if (t.r != prev_r || t.c != prev_c) {
                mat.row_ptr[t.r + 1]++;
                prev_r = t.r; prev_c = t.c;
            }
        }
    }
    for (int i = 1; i <= M; ++i) mat.row_ptr[i] += mat.row_ptr[i - 1];

    int unique_nnz = mat.row_ptr[M];
    mat.values.resize(unique_nnz, 0.0);
    mat.col_idx.resize(unique_nnz, 0);
    mat.nnz = unique_nnz;

    // Pass 2: fill, merging duplicates
    {
        std::vector<int> pos(mat.row_ptr.begin(), mat.row_ptr.end() - 1);
        int prev_r = -1, prev_c = -1, cur_idx = -1;
        for (auto& t : triplets) {
            if (t.r != prev_r || t.c != prev_c) {
                cur_idx = pos[t.r]++;
                mat.col_idx[cur_idx] = t.c;
                mat.values[cur_idx]  = t.v;
                prev_r = t.r; prev_c = t.c;
            } else {
                mat.values[cur_idx] += t.v;  // merge duplicate
            }
        }
    }

    return mat;
}

// ---------------------------------------------------------------------------
// SpMV  y = A * x
// ---------------------------------------------------------------------------
inline void spmv(const CSRMatrix& A,
                 const std::vector<double>& x,
                       std::vector<double>& y)
{
    if (static_cast<int>(x.size()) != A.cols)
        throw std::runtime_error("spmv: x size != A.cols");
    y.assign(A.rows, 0.0);
    for (int r = 0; r < A.rows; ++r) {
        double s = 0.0;
        for (int j = A.row_ptr[r]; j < A.row_ptr[r + 1]; ++j)
            s += A.values[j] * x[A.col_idx[j]];
        y[r] = s;
    }
}

// ---------------------------------------------------------------------------
// Convenience: wrap CSRMatrix into the Matvec functor expected by
// gmres::solve() and ca_gmres::solve()
//
// Usage:
//   CSRMatrix A = load_matrix_market("problem.mtx");
//   auto mv = spmv_matvec(A);
//   gmres::solve(mv, b, x);
// ---------------------------------------------------------------------------
inline std::function<void(const std::vector<double>&, std::vector<double>&)>
spmv_matvec(const CSRMatrix& A)
{
    return [&A](const std::vector<double>& x, std::vector<double>& y) {
        spmv(A, x, y);
    };
}