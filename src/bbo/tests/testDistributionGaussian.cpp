/**
 * \file testDistributionGaussian.cpp
 * \author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
 * 
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#include "bbo/DistributionGaussian.hpp"

#include <fstream>
#include <sstream>

#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;
using namespace DmpBbo;

int main(int n_args, char* args[])
{
  // If program has an argument, it is a directory to which to save files too (or --help)
  string directory;
  if (n_args>1)
    directory = args[1];
    
  int dim = 2;
  VectorXd mean(dim);
  mean << 0.0, 1.0;
  MatrixXd covar(dim,dim);
  covar << 3.0, 0.5, 0.5, 1.0;
  MatrixXd covar_diag(dim,dim);
  covar_diag << 3.0, 0.0, 0.0, 1.0;
  
  DistributionGaussian distribution1(mean, covar); 
  DistributionGaussian distribution2(mean, covar); 
  
  
  // Just to check if they generate different numbers
  int n_samples = 3;
  MatrixXd samples(n_samples,dim);
  distribution1.generateSamples(n_samples, samples);
  cout << "________________" << endl;
  cout << distribution1 << endl;
  cout << "samples distribution 1 =" << endl << samples << endl;
  cout << "________________" << endl;
  cout << distribution2 << endl;
  cout << "samples distribution 2 =" << endl << samples << endl;
  
  
  if (directory.empty())
  {
    //cout << "________________\nsamples =\n" << samples << endl;
  } 
  else 
  {
    n_samples = 1000;
    distribution1.generateSamples(n_samples, samples);
  
    ofstream outfile;
    string filename = directory+"/samples.txt";
    outfile.open(filename.c_str()); 
    if (!outfile.is_open())
    {
      cerr << __FILE__ << ":" << __LINE__ << ":";
      cerr << "Could not open file " << filename << " for writing." << endl;
    } 
    else
    {
      outfile << samples;
      outfile.close();
    }
  }

}


