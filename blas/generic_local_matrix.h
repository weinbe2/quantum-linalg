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

// Perform a local square transpose.
template<typename T> inline void cMATtranspose_square_local(T* mat, int ndim)
{
  for (int i = 0; i < ndim; i++)
    for (int j = i+1; j < ndim; j++)
      std::swap(mat[i*ndim+j], mat[j*ndim+i]);
}

// Perform a copy and transpose.
template<typename T> inline void cMATcopy_transpose_square_local(T* mat, T* mat_dest, int ndim)
{
  for (int i = 0; i < ndim; i++)
    for (int j = 0; j < ndim; j++)
      mat_dest[i*ndim+j] = mat[j*ndim+i];
}

// Perform a local hermitian transpose.
template<typename T> inline void cMATconjtrans_square_local(complex<T>* mat, int ndim)
{
  complex<T> tmp; 
  for (int i = 0; i < ndim; i++)
  {
    tmp = mat[i*ndim+i];
    mat[i*ndim+i] = std::conj(tmp);
    for (int j = i+1; j < ndim; j++)
    {
      tmp = std::conj(mat[i*ndim+j]);
      mat[i*ndim+j] = std::conj(mat[j*ndim+i]);
      mat[j*ndim+i] = tmp;
    }
  }
}

// Perform a copy and hermitian conjugate
template<typename T> inline void cMATcopy_conjtrans_square_local(T* mat, T* mat_dest, int ndim)
{
  for (int i = 0; i < ndim; i++)
    for (int j = 0; j < ndim; j++)
      mat_dest[i*ndim+j] = std::conj(mat[j*ndim+i]);
}



#endif // QLINALG_MATRIX_LOCAL