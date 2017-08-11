#include <iostream>
#include <complex>
#include <cmath>
#include <random>
#include <vector>

using namespace std;


// Random gaussian vector, random in real and imag.
template <typename T> inline void gaussian(complex<T>* x, int size, std::mt19937 &generator, T deviation = 1.0, complex<T> mean = 0.0)
{
  // Generate a normal distribution.
  std::normal_distribution<> dist(0.0, deviation);
  for (int i = 0; i < size; i++)
  {
    x[i] = std::complex<T>(real(mean) + static_cast<T>(dist(generator)), imag(mean) + static_cast<T>(dist(generator)));
  }
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

// Rescale cax, x *= a
template<typename T, typename U = T> inline void cax(U a, T* x, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] *= a;
  }
}

// Implement caxpy, y += a*x
template<typename T, typename U = T> inline void caxpy(U a, T* x, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += a*x[i];
  }
}

int main(int argc, char** argv)
{
  // Iterators
  int i, j;

  // Random number generator
  std::mt19937 generator (1337u);  

  // How many vectors are we bi-orthogonalizing?
  const int n_ortho = 5;

  // What's the length of these vectors?
  const int length = 1024;

  // Populate some vectors.
  vector<complex<double>*> left_vecs(n_ortho);
  vector<complex<double>*> right_vecs(n_ortho);
  for (i = 0; i < n_ortho; i++)
  {
    left_vecs[i] = new complex<double>[length];
    right_vecs[i] = new complex<double>[length];

    gaussian(left_vecs[i], length, generator);
    gaussian(right_vecs[i], length, generator);
  }

  // Print <l_i, r_j> for now.
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      std::cout << dot(left_vecs[i], right_vecs[j], length) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Okay, now bi-ortho.
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < i; j++)
    {
      // divisor would be one if we normalize.
      complex<double> alpha = dot(left_vecs[i], right_vecs[j], length)/dot(left_vecs[j], right_vecs[j], length);
      caxpy(-conj(alpha), left_vecs[j], left_vecs[i], length);
      complex<double> beta = dot(left_vecs[j], right_vecs[i], length)/dot(left_vecs[j], right_vecs[j], length);
      caxpy(-beta, right_vecs[j], right_vecs[i], length);
    }

    // We have a freedom on how to normalize since we need to take care of the 
    // magnitude and the phase. As a convention we're sticking it on the left.
    complex<double> cplx_dot = dot(left_vecs[i], right_vecs[i], length);
    double cplx_norm = abs(cplx_dot);
    cax(polar(1.0/sqrt(cplx_norm), arg(cplx_dot)), left_vecs[i], length);
    cax(1.0/sqrt(cplx_norm), right_vecs[i], length);
  }

  // Test printing it again.
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      std::cout << dot(left_vecs[i], right_vecs[j], length) << " ";
    }
    std::cout << "\n";
  }

  for (i = 0; i < n_ortho; i++)
  {
    delete[] left_vecs[i];
    delete[] right_vecs[i];
  }

  return 0;
}
