// Copyright (c) 2017 Evan S Weinberg
// Header file for templated vector operations.

#include <complex>
#include <random>

using std::complex; 
using std::polar;

#ifndef GENERIC_VECTOR
#define GENERIC_VECTOR

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Create and destroy a vector.
template<typename T> inline T* allocate_vector(int size)
{
  return new T[size];
}

template<typename T> inline void deallocate_vector(T** x)
{
  delete[] *x;
  *x = 0;
}

// Zeroes a vector.
template<typename T> inline void zero(T* x, int size)
{
   for (int i = 0; i < size; i++)
   {
      x[i] = 0.0;
   }
}

template <typename T> inline void zero(complex<T>* x, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = 0.0;
  }
}

// Assign a vector to a constant everywhere.
template <typename T> inline void constant(T* x, T val, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = val;
  }
}

template <typename T> inline void constant(complex<T>* x, T val, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = val;
  }
}

template <typename T> inline void constant(complex<T>* x, complex<T> val, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = val;
  }
}

// Random gaussian vector.
template<typename T> inline void gaussian(T* x, int size, std::mt19937 &generator, double deviation = 1.0)
{
  // Generate a normal distribution.
  std::normal_distribution<> dist(0.0, deviation);
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<T>(dist(generator));
  }
}

// Random gaussian vector, random in real and imag.
template <typename T> inline void gaussian(complex<T>* x, int size, std::mt19937 &generator, double deviation = 1.0)
{
  // Generate a normal distribution.
  std::normal_distribution<> dist(0.0, deviation);
  for (int i = 0; i < size; i++)
  {
    x[i] = std::complex<T>(static_cast<T>(dist(generator)), static_cast<T>(dist(generator)));
  }
}

// random vector on a uniform interval.
template <typename T> inline void random_uniform(T* x, int size, std::mt19937 &generator, T min, T max)
{
  std::uniform_real_distribution<> dist(min, max);
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<T>(dist(generator));
  }
}

template <typename T> inline void random_uniform(complex<T>* x, int size, std::mt19937 &generator, T min, T max)
{
  std::uniform_real_distribution<> dist(min, max);
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<complex<T>>(dist(generator));
  }
}

// vectorized polar. applies x = polar(1.0, real(x));. Ignores imag(x).
template <typename T> inline void polar(complex<T>* x, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<complex<T>>(polar(1.0, real(x[i])));
  }
}

// vectorized arg.
template <typename T> inline void arg(complex<T>* x, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<complex<T>>(arg(x[i]));
  }
}

// Copy v2 into v1.
template<typename T> inline void copy(T* v1, T* v2, int size)
{
  for (int i = 0; i < size; i++)
  {
    v1[i] = v2[i];
  }
}

template<typename T> inline void copy(complex<T>* v1, complex<T>* v2, int size)
{
  for (int i = 0; i < size; i++)
  {
    v1[i] = v2[i];
  }
}

// Implement cxpy, y += x.
template<typename T> inline void cxpy(T* x, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += x[i];
  }
}

// Implement cxpy, v2 += v1.
template<typename T> inline void cxpy(complex<T>* x, complex<T>* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += x[i];
  }
}

// Implement caxpy, y += a*x
template<typename T> inline void caxpy(T a, T* x, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += a*x[i];
  }
}

template<typename T> inline void caxpy(T a, complex<T>* x, complex<T>* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += a*x[i];
  }
}

template<typename T> inline void caxpy(complex<T> a, complex<T>* x, complex<T>* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += a*x[i];
  }
}

// Implement cxpay, y = x + a*y
template<typename T> inline void cxpay(T* x, T a, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] = x[i] + a*y[i];
  }
}

template<typename T> inline void cxpay(complex<T>* x, T a, complex<T>* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] = x[i] + a*y[i];
  }
}

template<typename T> inline void cxpay(complex<T>* x, complex<T> a, complex<T>* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] = x[i] + a*y[i];
  }
}

// Implement cxpayz, z = x + a*y
template<typename T> inline void cxpayz(T* x, T a, T* y, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] = x[i] + a*y[i];
  }
}

template<typename T> inline void cxpayz(complex<T>* x, T a, complex<T>* y, complex<T>* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] = x[i] + a*y[i];
  }
}

template<typename T> inline void cxpayz(complex<T>* x, complex<T> a, complex<T>* y, complex<T>* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] = x[i] + a*y[i];
  }
}

// Implement caxpbypz, z += a*x + b*y
template<typename T> inline void caxpbypz(T a, T* x, T b, T* y, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] += a*x[i] + b*y[i];
  }
}

template<typename T> inline void caxpbypz(T a, complex<T>* x, T b, complex<T>* y, complex<T>* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] += a*x[i] + b*y[i];
  }
}
template<typename T> inline void caxpbypz(complex<T> a, complex<T>* x, complex<T> b, complex<T>* y, complex<T>* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] += a*x[i] + b*y[i];
  }
}

// Implement caxpbypcz, z = a*x + b*y + cz
template<typename T> inline void caxpbypcz(T a, T* x, T b, T* y, T c, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] = a*x[i] + b*y[i] + c*z[i];
  }
}

template<typename T> inline void caxpbypcz(T a, complex<T>* x, T b, complex<T>* y, T c, complex<T>* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] = a*x[i] + b*y[i] + c*z[i];
  }
}

template<typename T> inline void caxpbypcz(complex<T> a, complex<T>* x, complex<T> b, complex<T>* y, complex<T> c, complex<T>* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] = a*x[i] + b*y[i] + c*z[i];
  }
}

// Computes v1 dot v2.
template<typename T> inline T dot(T* v1, T* v2, int size)
{
  T res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + v1[i]*v2[i];
  }
  return res;
}

// computes conj(v1) dot v2.
template <typename T> inline complex<T> dot(complex<T>* v1, complex<T>* v2, int size)
{
  complex<T> res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + conj(v1[i])*v2[i];
  }
  return res;
}

// Computes the vector norm squared.
template <typename T>
inline T norm2sq(T* v1, int size)
{
  T res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + v1[i]*v1[i];
  }
  return res;
}

template <typename T>
inline T norm2sq(complex<T>* v1, int size)
{
  T res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + real(conj(v1[i])*v1[i]);
  }
  return res;
}

// Return |v1 - v2|^2
template <typename T>
inline T diffnorm2sq(T* v1, T* v2, int size)
{
    T res = static_cast<T>(0.0);
    for (int i = 0; i < size; i++)
    {
        res = res + (v1[i] - v2[i])*(v1[i] - v2[i]);
    }
    return res;
}

template <typename T>
inline T diffnorm2sq(complex<T>* v1, complex<T>* v2, int size)
{
    T res = static_cast<T>(0.0);
    for (int i = 0; i < size; i++)
    {
        res = res + real(conj(v1[i] - v2[i])*(v1[i] - v2[i]));
    }
    return res;
}
    

template <typename T>
inline void normalize(T* v1, int size)
{
    T res = 1.0/sqrt(norm2sq<T>(v1, size));
    if (res > 0.0)
    {
        for (int i = 0; i < size; i++)
        {
            v1[i] *= res;
        }
    }
}

template <typename T>
inline void normalize(complex<T>* v1, int size)
{
    T res = 1.0/sqrt(norm2sq<T>(v1, size));
    if (res > 0.0)
    {
        for (int i = 0; i < size; i++)
        {
            v1[i] *= res;
        }
    }
}

template <typename T>
inline void conj(T* v1, int size)
{
    return; // Trivial, it's real.
}

template <typename T>
inline void conj(complex<T>* v1, int size)
{
    for (int i = 0; i < size; i++)
    {
        v1[i] = conj(v1[i]);
    }
}

// Make vector v1 orthogonal to vector v2. 
template <typename T>
inline void orthogonal(T* v1, T* v2, int size)
{
    T alpha = -dot<T>(v2, v1, size)/norm2sq<T>(v2, size);   
    for (int i = 0; i < size; i++)
    {
        v1[i] = v1[i] + alpha*v2[i];
    }
}

// Make vector v1 orthogonal to vector v2. 
template <typename T>
inline void orthogonal(complex<T>* v1, complex<T>* v2, int size)
{
    complex<T> alpha = -dot<T>(v2, v1, size)/norm2sq<T>(v2, size);   
    for (int i = 0; i < size; i++)
    {
        v1[i] = v1[i] + alpha*v2[i];
    }
}

#endif // GENERIC_VECTOR

