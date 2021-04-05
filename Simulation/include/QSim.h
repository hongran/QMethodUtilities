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
//#include "AnalysisModuleBase.h"
#include "AnalysisModuleList.h"

// Qmethod parameters
#define TIMEDECIMATION 60 // 2018 production run

// simulator parameters
#define NEMAX 1000 // max electrons per fill per calo
#define NTMAX 6    // max threshold histograms

//constants
//const int nsPerFill = 337500;
const int nsPerFill = 225000;
const float nsPerTick = 1000./800.;
const int qBinSize = TIMEDECIMATION*nsPerTick;
// const double GeVToADC = 1013.0*10; //The 10 is tuned by Ran Hong
const double GeVToADC = 1013.0; //Remove 10 by Fang Han, fang.han@uky.edu
const double PeakBinFrac = 0.4;

const int nxseg = 9;
const int nyseg = 6;
const int nsegs = nxseg*nyseg;

namespace QSimulation{
  class QSim{
    public:
      QSim(const std::map<std::string,int>& tIntParameters,const std::map<std::string,float> & tFloatParameters, const std::map<std::string,std::string> & tStringParameters, long long int tSeed = 0);
      ~QSim();
      int Simulate(int NFlushes);
      int Output(int RunNumber);
      int GetArray(std::string ArrayName,std::vector<double>& Output);
      int GetCaloArray(std::string ArrayName,std::vector<double>& Output, bool BatchSum = true);
      
      //Set Functions
      std::vector<float> GetIntegratedPulseTemplate(){return IntegratedPulseTemplate;}
      std::vector<float> GetPedestalTemplate(){return PedestalTemplate;}
      int RegisterAnalysisModule(std::string ModuleName, const std::map<std::string,int>& tIntParameters,const std::map<std::string,float> & tFloatParameters,const std::map<std::string,std::string> & tStringParameters, int nFlushesPerBatch, int FillMaxLength);
      
    private:
      //Parameter Map, arrays and device arrays
      std::map<std::string,int> IntParameters;
      std::map<std::string,float> FloatParameters;
      std::map<std::string,std::string> StringParameters;

      std::vector<float> SimulationParameters;
      std::vector<int> SimulationIntParameters;

      float * d_SimulationParameters;
      int * d_SimulationIntParameters;

      float * d_IntegratedPulseTemplate;
      float * d_PedestalTemplate;

      // for state of randum generators
      curandState *d_state;

      //Pulse Shape
      std::vector<float> IntegratedPulseTemplate;
      std::vector<float> PedestalTemplate;

      // Q-method arrays
      std::map<std::string,float *> HostArrays;
      std::map<std::string,float *> DeviceArrays;
      std::map<std::string,int> ArraySizes;

      // Analysis Module array
      std::map<std::string,QAnalysis::AnalysisModule *> AnaModules;

      // for energy array
      //int32_t *h_energyArray, *d_energyArray;
      //float *h_energySumArray, *d_energySumArray;

      //Private methods
      int InitParameters();
      int IntegratePulseTemplate(std::string TemplatePath,int CrystalId,int TemplateSize,int TemplateZero);
      int LoadPedestalTemplate(std::string TemplatePath,int CrystalId);
  };
}

#endif
