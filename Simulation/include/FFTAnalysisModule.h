#ifndef ANALYSIS_MODULE_FFT_H
#define ANALYSIS_MODULE_FFT_H

#include "AnalysisModuleBase.h"

#include <map>
#include <string>
#include <vector>

// Cuda include
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <curand.h>
#include <curand_kernel.h>
// #include <helper_cuda.h>

namespace QAnalysis {
class FFTAnalysisModule : public AnalysisModule {
public:
  FFTAnalysisModule(std::string Name,
                    const std::map<std::string, int> &tIntParameters,
                    const std::map<std::string, float> &tFloatParameters,
                    const std::map<std::string, std::string> &tStringParameters,
                    int nFlushesPerBatch, int FillMaxLength);

  int FlushAnalysis(std::map<std::string, float *> *SimulatorHostArrays,
                    std::map<std::string, float *> *SimulatorDeviceArrays,
                    std::map<std::string, int> *SimulatorArraySizes) override;
  int EndAnalysis(std::map<std::string, float *> *SimulatorHostArrays,
                  std::map<std::string, int> *SimulatorArraySizes) override;
  int Output(int RunNumber, int nFlush) override;

  int GetArray(std::string ArrayName, std::vector<double> &Output);
  int GetCaloArray(std::string ArrayName, std::vector<double> &Output,
                   bool BatchSum = true);

  cufftHandle fftPlan;
  std::map<std::string, cufftComplex *> FFTArrays;

protected:
  int InitParameters();
};
} // end namespace QAnalysis

#endif
