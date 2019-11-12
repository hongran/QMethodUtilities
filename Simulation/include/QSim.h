#ifndef QSIM_H
#define QSIM_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <time.h>
#include <math.h>
#include <memory>

// CUDA includes
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// calo parameters
#define NXSEG 9
#define NYSEG 6
#define NSEG 54

// Qmethod parameters
#define TIMEDECIMATION 60 // 2018 production run

// simulator parameters
#define NEMAX 5000 // max electrons per fill per calo
#define NTMAX 6    // max threshold histograms

//constants
const int nsPerFill = 337500;
const int nsPerTick = 1000./800.;
const int qBinSize = TIMEDECIMATION*nsPerTick;
const double GeVToADC = 1013.0;
const double PeakBinFrac = 0.4;

const int nxseg = 9;
const int nyseg = 6;
const int nsegs = nxseg*nyseg;

namespace QSimulation{
  class QSim{
    public:
      QSim(int t_nthreads,int t_nfills,int t_ne,int t_nbatches,float t_threshold,int t_window,bool t_fillnoise,bool t_flashgainsag);
      ~QSim();
      int Simulate(int NFlushes);
      int GetArray(std::string ArrayName,std::vector<double>& Output);
    private:
      // define nthreads per block for GPU
      int nthreads; //number of threads per block
      int nfills; //number of fills per flush
      int ne; // number of electrons per fill
      int nbatches; //number of batches of fills for each GPU computing cycle

      int fill_buffer_max_length; // fill length in unit of hostogram bins

      float threshold;
      int window;
      bool fillnoise;
      bool flashgainsag;

      // for state of randum generators
      curandState *d_state;

      // Q-method arrays
      std::map<std::string,float *> HostArrays;
      std::map<std::string,float *> DeviceArrays;
      std::map<std::string,int> ArraySizes;

      // for energy array
      //int32_t *h_energyArray, *d_energyArray;
      //float *h_energySumArray, *d_energySumArray;
  };
}

#endif
