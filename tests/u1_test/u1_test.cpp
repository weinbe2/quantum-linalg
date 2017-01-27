// Test the various U1 routines in u1_utils.cpp.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <random>

using namespace std;

#include "u1_utils.h"

int main(int argc, char** argv)
{
  cout << setiosflags(ios::fixed) << setprecision(6);

  // Some basic fields.
  const int x_len = 16;
  const int y_len = 16;
  const int volume = x_len*y_len;
  const int gauge_volume = 2*volume;
  double beta = 3.0;
  double alpha = 0.0; 
  int n_iter = 1; 

  std::mt19937 generator (1337u);

  int i, j, x, y;

  // Two gauge fields.
  complex<double>* field1 = new complex<double>[gauge_volume];
  complex<double>* field2 = new complex<double>[gauge_volume];

  // One gauge transform.
  complex<double>* trans1 = new complex<double>[volume];

  // Create a unit field!
  unit_gauge_u1(field1, x_len, y_len);

  cout << "A unit gauge field has average plaquette " << get_plaquette_u1(field1, x_len, y_len) << "\n";

  // Create a series of random fields. 
  for (beta = 100; beta > 1e-4; beta*=0.1)
  {
    gauss_gauge_u1(field1, x_len, y_len, generator, beta);
    cout << "A gauge field with beta " << beta << " has average plaquette " << get_plaquette_u1(field1, x_len, y_len);

    // Smear the field.
    apply_ape_smear_u1(field2, field1, x_len, y_len, alpha, n_iter);
    cout << " and, after " << n_iter << " iteration(s) of ape smearing with alpha=" << alpha << ", has plaquette " << get_plaquette_u1(field2, x_len, y_len) << " and topology " << get_topo_u1(field2, x_len, y_len) << "\n"; 
  }

  // Create a hot field.
  rand_gauge_u1(field1, x_len, y_len, generator);
  cout << "A random gauge field has average plaquette " << get_plaquette_u1(field1, x_len, y_len) << "\n";

  // Save the gauge field to file.
  cout << "Saving the random gauge field to file...\n";
  write_gauge_u1(field1, x_len, y_len, "./cfg/cfg16_hot.dat");

  // Load the gauge field.
  cout << "Load the random gauge field from file...\n";
  read_gauge_u1(field2, x_len, y_len, "./cfg/cfg16_hot.dat");

  // Check the plaquette.
  cout << "The loaded gauge field has average plaquette " << get_plaquette_u1(field2, x_len, y_len) << "\n";

  // Get a random gauge transform.
  cout << "Getting a random gauge transform.\n";
  rand_trans_u1(trans1, x_len, y_len, generator);

  // Apply the random gauge transform, check plaquette.
  cout << "Applying the random gauge transform.\n";
  apply_gauge_trans_u1(field2, trans1, x_len, y_len);

  cout << "The average plaquette after a random gauge transform is " << get_plaquette_u1(field2, x_len, y_len) << "\n";

  // Clean up.
  delete[] field1;
  delete[] field2;
  delete[] trans1;
}

