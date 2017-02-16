// Copyright (c) 2017 Evan S Weinberg
// Header file for templated local matrix operations.

#include <complex>

using std::complex;

#ifndef QLINALG_MATRIX_LOCAL
#define QLINALG_MATRIX_LOCAL

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Do a local mat-vec operation in row-major.
// I should just pull in Eigen for this,
// but why overcomplicate things for now.
// y += A*x, A matrix. y is length nrow, x is length ncol.
template<typename T> inline void cMATxpy_local(T* mat, T* x, T* y, int nrow, int ncol)
{
  for (int i = 0; i < nrow; i++)
  {
    T tmp = static_cast<T>(0.0);
    for (int j = 0; j < ncol; j++)
      tmp += mat[i*ncol+j]*x[j];
    y[i] += tmp;
  }
}





#endif // QLINALG_MATRIX_LOCAL