#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TH1.h"

#include "TruthAnalysisModule.h"

// Analysis device kernel
__global__ void flush_analysis(float *FlushQTruthArray, float *AnaQArray,
                               int *AnalysisIntParameters,
                               float *AnalysisParameters) {
  int flush_buffer_max_length = AnalysisIntParameters[1];
  int nFlushesPerBatch = AnalysisIntParameters[0];
  float threshold = AnalysisParameters[0];

  // thread index
  int iflush = blockIdx.x * blockDim.x + threadIdx.x;

  if (iflush < nFlushesPerBatch) {
    for (int idx = 0; idx < NSEG * flush_buffer_max_length; idx++) {
      int flushoffset = iflush * NSEG * flush_buffer_max_length;
      float qdata= FlushQTruthArray[flushoffset + idx];
      if(qdata > threshold){
        AnaQArray[flushoffset + idx] += qdata;
      }
      /*
      if (iflush > 20000)
      {
	printf("%d\n",idx);
      }
      */
    }
  }
}

namespace QAnalysis {
TruthAnalysisModule::TruthAnalysisModule(
    std::string Name, const std::map<std::string, int> &tIntParameters,
    const std::map<std::string, float> &tFloatParameters,
    const std::map<std::string, std::string> &tStringParameters,
    int nFlushesPerBatch, int FillMaxLength)
    : AnalysisModule(Name, tIntParameters, tFloatParameters,
                     tStringParameters) {
  // Initialize the parameter arrays
  IntParameters["NFlushesPerBatch"] = nFlushesPerBatch;
  IntParameters["FillBufferMaxLength"] = FillMaxLength;
  InitParameters();
  // cudaSetDevice(1);

  // Allocate Derive memory for parameters

  cudaMalloc((void **)&d_AnalysisParameters,
             AnalysisParameters.size() * sizeof(float));
  cudaMalloc((void **)&d_AnalysisIntParameters,
             AnalysisIntParameters.size() * sizeof(int));

  cudaMemcpy(d_AnalysisParameters, &AnalysisParameters[0],
             AnalysisParameters.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_AnalysisIntParameters, &AnalysisIntParameters[0],
             AnalysisIntParameters.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  // Arrays
  ArraySizes["AnaQArray"] =
      nFlushesPerBatch * NSEG * FillMaxLength * sizeof(float);

  // Allocate memories
  for (auto it = ArraySizes.begin(); it != ArraySizes.end(); ++it) {
    auto Name = it->first;
    auto Size = it->second;
    HostArrays[Name] = (float *)malloc(Size);
    cudaMalloc((void **)&DeviceArrays[Name], Size);
  }
}

// Analysis Functions
int TruthAnalysisModule::FlushAnalysis(
    std::map<std::string, float *> *SimulatorHostArrays,
    std::map<std::string, float *> *SimulatorDeviceArrays,
    std::map<std::string, int> *SimulatorArraySizes) {

  int nblocks =
      IntParameters["NFlushesPerBatch"] / IntParameters["NThreadsPerBlock"] + 1;
  std::cout << "Analyzing flush batch" << std::endl;

  flush_analysis<<<nblocks, IntParameters["NThreadsPerBlock"]>>>(
      (*SimulatorDeviceArrays)["FlushQTruthArray"], DeviceArrays["AnaQArray"],
      d_AnalysisIntParameters, d_AnalysisParameters);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Cuda failure with user kernel function make)flush anlaysis %s:%d: "
           "'%s'\n",
           __FILE__, __LINE__, cudaGetErrorString(err));
    exit(0);
  }

  return 0;
}

int TruthAnalysisModule::EndAnalysis(
    std::map<std::string, float *> *SimulatorHostArrays,
    std::map<std::string, int> *SimulatorArraySizes) {
  // copy back to host
  int n = 0;
  cudaError err;
  for (auto it = ArraySizes.begin(); it != ArraySizes.end(); ++it) {
    auto Name = it->first;
    auto Size = it->second;
    cudaMemcpy(HostArrays[Name], DeviceArrays[Name], Size,
               cudaMemcpyDeviceToHost);
    //      std::cout<< n << " "<<Name<<" "<<Size<<std::endl;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Cuda failure with user kernel function make)analysis endrun "
             "copy "
             "%s:%d: '%s'\n",
             __FILE__, __LINE__, cudaGetErrorString(err));
      exit(0);
    }
    n++;
  }

  return 0;
}

int TruthAnalysisModule::Output(int RunNumber) {
  std::vector<double> AnaQHist;
  this->GetCaloArray("AnaQArray", AnaQHist);

  unsigned int N = AnaQHist.size();

  TH1 *hAnaQ = new TH1D("TruthQHist", "TruthQHist", N, 0, N * 0.075);
  for (unsigned int i = 0; i < N; i++) {
    hAnaQ->SetBinContent(i, AnaQHist[i]);
  }


  TFile *FileOut =
      new TFile(Form("TruthRootOut_%04d.root", RunNumber), "recreate");
  hAnaQ->Write();
  FileOut->Close();

  return 0;
}

// Private Functions
int TruthAnalysisModule::InitParameters() {
  AnalysisParameters.resize(1);
  AnalysisIntParameters.resize(2);

  AnalysisParameters[0] = FloatParameters["Threshold"];
  AnalysisIntParameters[0] = IntParameters["NFlushesPerBatch"];
  AnalysisIntParameters[1] = IntParameters["FillBufferMaxLength"];
  // AnalysisIntParameters[2] = IntParameters["Window"];
  // AnalysisIntParameters[3] = IntParameters["Gap"];

  return 0;
}

int TruthAnalysisModule::GetArray(std::string ArrayName,
                               std::vector<double> &Output) {
  auto Size = ArraySizes[ArrayName];
  Output.resize(Size);
  auto ptr = HostArrays[ArrayName];
  for (int i = 0; i < Size; i++) {
    Output[i] = ptr[i];
  }
  return 0;
}

int TruthAnalysisModule::GetCaloArray(std::string ArrayName,
                                   std::vector<double> &Output, bool BatchSum) {
  Output.clear();
  Output.resize(IntParameters["FillBufferMaxLength"], 0.0);
  auto ptr = HostArrays[ArrayName];

  int nFlushesPerBatch = IntParameters["NFlushesPerBatch"];

  for (unsigned int k = 0; k < nFlushesPerBatch; k++) {
    for (unsigned int j = 0; j < NSEG; j++) {
      for (unsigned int i = 0; i < IntParameters["FillBufferMaxLength"]; i++) {
        Output[i] +=
            ptr[(k * NSEG + j) * IntParameters["FillBufferMaxLength"] + i];
      }
    }
    if (!BatchSum) {
      break;
    }
  }

  return 0;
}

} // end namespace QAnalysis
