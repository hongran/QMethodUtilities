#ifndef QSIM_H
#define QSIM_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <time.h>
#include <math.h>
#include <memory>

// CUDA includes
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Global.h"

// calo parameters
#define NXSEG 9
#define NYSEG 6
#define NSEG 54

// Qmethod parameters
#define TIMEDECIMATION 60 // 2018 production run

// simulator parameters
#define NEMAX 1000 // max electrons per fill per calo
#define NTMAX 6    // max threshold histograms

//constants
const int nsPerFill = 337500;
const float nsPerTick = 1000./800.;
const int qBinSize = TIMEDECIMATION*nsPerTick;
const double GeVToADC = 1013.0*10; //The 10 is tuned by Ran Hong
const double PeakBinFrac = 0.4;

const int nxseg = 9;
const int nyseg = 6;
const int nsegs = nxseg*nyseg;

namespace QSimulation{
  class QSim{
    public:
      QSim(const std::map<std::string,float> & tFloatParameters, const std::map<std::string,int>& tIntParameters,long long int tSeed = 0);
      ~QSim();
      int Simulate(int NFlushes);
      int GetArray(std::string ArrayName,std::vector<double>& Output);
      int GetCaloArray(std::string ArrayName,std::vector<double>& Output, bool BatchSum = true);
      
      //Set Functions
      int SetIntegratedPulseTemplate(std::vector<float> temp,int Size,int ZeroIndex);
      int SetPedestalTemplate(std::vector<float> temp);
      
    private:
      //Parameter Map, arrays and device arrays
      std::map<std::string,float> FloatParameters;
      std::map<std::string,int> IntParameters;

      std::vector<float> SimulationParameters;
      std::vector<int> SimulationIntParameters;
      std::vector<float> AnalysisParameters;
      std::vector<int> AnalysisIntParameters;

      float * d_SimulationParameters;
      int * d_SimulationIntParameters;
      float * d_AnalysisParameters;
      int * d_AnalysisIntParameters;

      // for state of randum generators
      curandState *d_state;

      //Pulse Shape
      std::vector<float> IntegratedPulseTemplate;
      std::vector<float> PedestalTemplate;

      // Q-method arrays
      std::map<std::string,float *> HostArrays;
      std::map<std::string,float *> DeviceArrays;
      std::map<std::string,int> ArraySizes;

      // for energy array
      //int32_t *h_energyArray, *d_energyArray;
      //float *h_energySumArray, *d_energySumArray;

      //Private methods
      int InitParameters();
  };
}

#endif
