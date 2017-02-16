// Copyright (c) 2017 Evan S Weinberg
// Header file for templated vector operations.

#include <complex>
#include <random>

using std::complex; 
using std::polar;

#include "generic_local_matrix.h"

#ifndef QLINALG_MATRIX
#define QLINALG_MATRIX

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Apply the global mat-vec operation in row-major.
typename<template T> inline void cMATxy(T* mat, T* x, T* y, int nelem, int nrow, int ncol)
{
  int mat_vol = nrow*ncol;
  for (i = 0; i < nelem; i++)
  {
    cMATxy(mat+i*mat_vol, x+i*ncol, y+i*nrow, nrow, ncol);
  }
}

#endif // QLINALG_MATRIX

