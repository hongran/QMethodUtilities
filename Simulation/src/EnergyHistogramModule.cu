#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TH1.h"

#include "EnergyHistogramModule.h"

// Analysis device kernel
__global__ void
flush_adc_histogram(float *FlushHitArray, float *FlushQTruthArray,
                    float *S_ADCHistArray, float *C_ADCHistArray,
                    int *AnalysisIntParameters, float *AnalysisParameters) {
  int flush_buffer_max_length = AnalysisIntParameters[1];
  int nFlushesPerBatch = AnalysisIntParameters[0];
  // Energy bin width.
  int EBinW = AnalysisIntParameters[2];
  // Low energy cut in energy in ADC histogram.
  int lowECut = AnalysisIntParameters[3];

  // thread index
  int iflush = blockIdx.x * blockDim.x + threadIdx.x;

  if (iflush < nFlushesPerBatch) {

    int flushOffset = iflush * NSEG * flush_buffer_max_length;
    for (int i = 0; i < flush_buffer_max_length; i++) {
      float sig_sum = 0;
      for (int iseg = 0; iseg < NSEG; iseg++) {
        int segOffset = iseg * flush_buffer_max_length;
        int index = flushOffset + segOffset + i;
        float signal = FlushQTruthArray[index];
        sig_sum += signal;
        uint s_binIdx = __float2uint_rd((signal - lowECut) / EBinW);
        atomicAdd(&S_ADCHistArray[s_binIdx], 1);
      }
      uint c_binIdx = __float2uint_rd((sig_sum - lowECut) / EBinW);
      atomicAdd(&C_ADCHistArray[c_binIdx], 1);
    }
  }
}

namespace QAnalysis {

EnergyHistogramModule::EnergyHistogramModule(
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
  ArraySizes["EnergyHistArray"] = 1000 * sizeof(float);
  ArraySizes["SegmADCHistArray"] = 1000 * sizeof(float);
  ArraySizes["CaloADCHistArray"] = 1000 * sizeof(float);

  // Allocate memories
  for (auto it = ArraySizes.begin(); it != ArraySizes.end(); ++it) {
    auto Name = it->first;
    auto Size = it->second;
    HostArrays[Name] = (float *)malloc(Size);
    cudaMalloc((void **)&DeviceArrays[Name], Size);
  }
}

// Analysis Functions
int EnergyHistogramModule::FlushAnalysis(
    std::map<std::string, float *> *SimulatorHostArrays,
    std::map<std::string, float *> *SimulatorDeviceArrays,
    std::map<std::string, int> *SimulatorArraySizes) {

  int nblocks =
      IntParameters["NFlushesPerBatch"] / IntParameters["NThreadsPerBlock"] + 1;
  std::cout << "Analyzing flush batch" << std::endl;
  // std::cout << "ADCBinWidth: " << AnalysisIntParameters[2]<<std::endl;
  // std::cout << "LowADCCut: " << AnalysisIntParameters[3]<<std::endl;
  flush_adc_histogram<<<nblocks, IntParameters["NThreadsPerBlock"]>>>(
      (*SimulatorDeviceArrays)["FlushHitArray"],
      (*SimulatorDeviceArrays)["FlushQTruthArray"],
      DeviceArrays["SegmADCHistArray"], DeviceArrays["CaloADCHistArray"],
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

int EnergyHistogramModule::EndAnalysis(
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
      printf("Cuda failure with user kernel function make)analysis endrun copy "
             "%s:%d: '%s'\n",
             __FILE__, __LINE__, cudaGetErrorString(err));
      exit(0);
    }
    n++;
  }

  return 0;
}

int EnergyHistogramModule::Output(int RunNumber) {

  auto ptr1 = HostArrays["SegmADCHistArray"];
  auto ptr2 = HostArrays["CaloADCHistArray"];
  auto Size = ArraySizes["SegmADCHistArray"] / sizeof(float);
  std::cout << "ADCBinWidth: " << AnalysisIntParameters[2] << std::endl;
  std::cout << "LowADCCut: " << AnalysisIntParameters[3] << std::endl;
  std::cout << "Size: " << Size << std::endl;
  int ADCBinW = IntParameters["ADCBinWidth"];
  int lowADCCut = IntParameters["LowADCCut"];

  TH1 *hSegmADC = new TH1D("Segm_ADC_Hist", "Segments ADC Distribution", Size,
                           lowADCCut, lowADCCut + Size * ADCBinW);
  for (unsigned int i = 0; i < Size; i++) {
    hSegmADC->SetBinContent(i, ptr1[i]);
  }
  TH1 *hCaloADC = new TH1D("Calo_ADC_Hist", "Calorimeter ADC Distribution",
                           Size, lowADCCut, lowADCCut + Size * ADCBinW);
  for (unsigned int i = 0; i < Size; i++) {
    hCaloADC->SetBinContent(i, ptr2[i]);
  }

  TFile *FileOut =
      new TFile(Form("Energy2ADCHist_%04d.root", RunNumber), "recreate");
  hSegmADC->Write();
  hCaloADC->Write();
  FileOut->Close();

  return 0;
}

// Private Functions
int EnergyHistogramModule::InitParameters() {
  AnalysisParameters.resize(1);
  AnalysisIntParameters.resize(4);

  AnalysisParameters[0] = FloatParameters["Threshold"];
  AnalysisIntParameters[0] = IntParameters["NFlushesPerBatch"];
  AnalysisIntParameters[1] = IntParameters["FillBufferMaxLength"];
  AnalysisIntParameters[2] = IntParameters["ADCBinWidth"];
  AnalysisIntParameters[3] = IntParameters["LowADCCut"];
  for(auto item : AnalysisIntParameters){
    std::cout<<"Parameter: "<<item <<std::endl;
  }
  return 0;
}

int EnergyHistogramModule::GetArray(std::string ArrayName,
                                    std::vector<double> &Output) {
  auto Size = ArraySizes[ArrayName];
  Output.resize(Size);
  auto ptr = HostArrays[ArrayName];
  for (int i = 0; i < Size; i++) {
    Output[i] = ptr[i];
  }
  return 0;
}

int EnergyHistogramModule::GetCaloArray(std::string ArrayName,
                                        std::vector<double> &Output,
                                        bool BatchSum) {
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
