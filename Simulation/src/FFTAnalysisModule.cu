#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TH1.h"

#include "FFTAnalysisModule.h"
#include <helper_cuda.h>

// Initialize complex fft array with real input
__global__ void init_fftArray(float *FlushQ, cufftComplex *FlushFFTQ,
                              int *AnalysisIntParameters) {
  int flush_buffer_max_length = AnalysisIntParameters[1];
  int nFlushesPerBatch = AnalysisIntParameters[0];

  // thread index
  int iflush = blockIdx.x * blockDim.x + threadIdx.x;

  if (iflush < nFlushesPerBatch) {
    for (int idx = 0; idx < NSEG * flush_buffer_max_length; idx++) {
      int flushoffset = iflush * NSEG * flush_buffer_max_length;
      FlushFFTQ[flushoffset + idx].x = FlushQ[flushoffset + idx];
      FlushFFTQ[flushoffset + idx].y = 0;
    }
  }
}
__global__ void apply_mask(cufftComplex *FlushFFTQ,
                           int *AnalysisIntParameters) {
  int flush_buffer_max_length = AnalysisIntParameters[1];
  int nFlushesPerBatch = AnalysisIntParameters[0];
  int lowN = AnalysisIntParameters[2];
  // thread index
  int iflush = blockIdx.x * blockDim.x + threadIdx.x;

  if (iflush < nFlushesPerBatch) {
    for (int itrace = 0; itrace < NSEG; itrace++) {
      int offset = (iflush * NSEG + itrace) * flush_buffer_max_length;
      for (int idx = 0; idx < lowN; idx++) {
        FlushFFTQ[offset + idx].x = 0;
        FlushFFTQ[offset + idx].y = 0;
      }
      for (int idx = flush_buffer_max_length - lowN;
           idx < flush_buffer_max_length; idx++) {
        FlushFFTQ[offset + idx].x = 0;
        FlushFFTQ[offset + idx].y = 0;
      }
    }
  }
}
// Sum FFT Analysis device kernel
__global__ void sum_flushes(float *FlushQ, cufftComplex *FlushFFTQ,
                            float *AnaQArray, float *AnaPedArray,
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
      auto signal =
          FlushFFTQ[flushoffset + idx].x / (float)flush_buffer_max_length;
      if (signal < threshold) continue;
      AnaQArray[flushoffset + idx] += signal;
      AnaPedArray[flushoffset + idx] += FlushQ[flushoffset + idx] - signal;
    }
  }
}
// Histogram FFT method processed ADCs
__global__ void
fft_adc_hist(cufftComplex *FFTQArray,
             float *S_ADCHistArray, float *C_ADCHistArray,
             int *AnalysisIntParameters, float *AnalysisParameters) 
{
  int flush_buffer_max_length = AnalysisIntParameters[1];
  int nFlushesPerBatch = AnalysisIntParameters[0];
  float threshold = AnalysisParameters[0];
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
        float signal = FFTQArray[index].x;
        if (signal < threshold) continue;
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
FFTAnalysisModule::FFTAnalysisModule(
    std::string Name, const std::map<std::string, int> &tIntParameters,
    const std::map<std::string, float> &tFloatParameters,
    const std::map<std::string, std::string> &tStringParameters,
    int nFlushesPerBatch, int FillMaxLength)
    : AnalysisModule(Name, tIntParameters, tFloatParameters,
                     tStringParameters) {

  std::cout<<"Constructing FFTAnalysisModule..."<<std::endl;
  // Initialize the parameter arrays
  IntParameters["NFlushesPerBatch"] = nFlushesPerBatch;
  IntParameters["FillBufferMaxLength"] = FillMaxLength;
  InitParameters();

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
  ArraySizes["AnaPedArray"] =
      nFlushesPerBatch * NSEG * FillMaxLength * sizeof(float);
  ArraySizes["FFTSegmADCHistArray"] = 1000 * sizeof(float);
  ArraySizes["FFTCaloADCHistArray"] = 1000 * sizeof(float);
  // Allocate memories
  for (auto it = ArraySizes.begin(); it != ArraySizes.end(); ++it) {
    auto Name = it->first;
    auto Size = it->second;
    HostArrays[Name] = (float *)malloc(Size);
    cudaMalloc((void **)&DeviceArrays[Name], Size);
  }
  // Allocate for FFT device memories
  int mem_size = sizeof(cufftComplex) * nFlushesPerBatch * NSEG * FillMaxLength;

  // Allocate device memory for signal
  checkCudaErrors(cudaMalloc((void **)&d_FFTArray, mem_size));

  int rank = 1;
  int idist = FillMaxLength;
  int odist = FillMaxLength;
  int nEl[1] = {FillMaxLength};
  int inembed[] = {0};
  int onembed[] = {0};
  int istride = 1;
  int ostride = 1;
  checkCudaErrors(cufftPlanMany(&fftPlan, rank, nEl, inembed, istride, idist,
                                onembed, ostride, odist, CUFFT_C2C,
                                nFlushesPerBatch * NSEG));
}

// Analysis Functions
int FFTAnalysisModule::FlushAnalysis(
    std::map<std::string, float *> *SimulatorHostArrays,
    std::map<std::string, float *> *SimulatorDeviceArrays,
    std::map<std::string, int> *SimulatorArraySizes) {

  int nblocks =
      IntParameters["NFlushesPerBatch"] / IntParameters["NThreadsPerBlock"] + 1;
  std::cout << "Analyzing flush batch" << std::endl;

  float time;
  cudaEvent_t start, stop;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  init_fftArray<<<nblocks, IntParameters["NThreadsPerBlock"]>>>(
      (*SimulatorDeviceArrays)["FlushQArray"], d_FFTArray,
      d_AnalysisIntParameters);

  checkCudaErrors(cufftExecC2C(fftPlan, d_FFTArray,
                               d_FFTArray, CUFFT_FORWARD));
  apply_mask<<<nblocks, IntParameters["NThreadsPerBlock"]>>>(
    d_FFTArray, d_AnalysisIntParameters);
  checkCudaErrors(cufftExecC2C(fftPlan, d_FFTArray,
    d_FFTArray, CUFFT_INVERSE));
  fft_adc_hist<<<nblocks, IntParameters["NThreadsPerBlock"]>>>(
    d_FFTArray,DeviceArrays["FFTSegmADCHistArray"], 
    DeviceArrays["FFTCaloADCHistArray"], d_AnalysisIntParameters, 
    d_AnalysisParameters);
  sum_flushes<<<nblocks, IntParameters["NThreadsPerBlock"]>>>(
      (*SimulatorDeviceArrays)["FlushQArray"], d_FFTArray,
      DeviceArrays["AnaQArray"], DeviceArrays["AnaPedArray"],
      d_AnalysisIntParameters, d_AnalysisParameters);

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

  printf("Time to Analyze flush:  %3.1f ms \n", time);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Cuda failure with user kernel function make flush anlaysis %s:%d: "
           "'%s'\n",
           __FILE__, __LINE__, cudaGetErrorString(err));
    exit(0);
  }

  return 0;
}

int FFTAnalysisModule::EndAnalysis(
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

int FFTAnalysisModule::Output(int RunNumber) {
  std::vector<double> AnaQHist;
  std::vector<double> AnaPedHist;
  this->GetCaloArray("AnaQArray", AnaQHist);
  this->GetCaloArray("AnaPedArray", AnaPedHist, false);

  unsigned int N = AnaQHist.size();

  TH1 *hAnaQ = new TH1D("FFTQHist", "FFTQHist", N, 0,
                        N * 0.075);
  for (unsigned int i = 0; i < N; i++) {
    hAnaQ->SetBinContent(i, AnaQHist[i]);
  }

  TH1 *hAnaPed = new TH1D("FFTPedHist", "FFTPedHist",
                          N, 0, N * 0.075);
  for (unsigned int i = 0; i < N; i++) {
    hAnaPed->SetBinContent(i, AnaPedHist[i]);
  }

  auto ptr1 = HostArrays["FFTSegmADCHistArray"];
  auto ptr2 = HostArrays["FFTCaloADCHistArray"];
  auto Size = ArraySizes["FFTSegmADCHistArray"] / sizeof(float);
  int ADCBinW = IntParameters["ADCBinWidth"];
  int lowADCCut = IntParameters["LowADCCut"];

  TH1 *hSegmADC = new TH1D("FFT_Segm_ADC_Hist", "Segments ADC Distribution, FFT Method", Size,
                           lowADCCut, lowADCCut + Size * ADCBinW);
  for (unsigned int i = 0; i < Size; i++) {
    hSegmADC->SetBinContent(i, ptr1[i]);
  }
  TH1 *hCaloADC = new TH1D("FFT_Calo_ADC_Hist", "Calorimeter ADC Distribution, FFT Method",
                           Size, lowADCCut, lowADCCut + Size * ADCBinW);
  for (unsigned int i = 0; i < Size; i++) {
    hCaloADC->SetBinContent(i, ptr2[i]);
  }

  TFile *FileOut =
      new TFile(Form("FFTRootOut_%04d.root", RunNumber), "recreate");
  hAnaQ->Write();
  hAnaPed->Write();
  hSegmADC->Write();
  hCaloADC->Write();
  FileOut->Close();

  return 0;
}

// Private Functions
int FFTAnalysisModule::InitParameters() {
  AnalysisParameters.resize(1);
  AnalysisIntParameters.resize(4);

  AnalysisParameters[0] = FloatParameters["Threshold"];

  AnalysisIntParameters[0] = IntParameters["NFlushesPerBatch"];
  AnalysisIntParameters[1] = IntParameters["FillBufferMaxLength"];
  AnalysisIntParameters[2] = IntParameters["LowN"];
  AnalysisIntParameters[3] = IntParameters["ADCBinWidth"];
  AnalysisIntParameters[4] = IntParameters["LowADCCut"];
  std::cout<<"Threshold: "<<AnalysisParameters[0]<<std::endl;
  std::cout<<"LowN: "<<AnalysisIntParameters[2]<<std::endl;
  std::cout<<"ADCBinWidth: "<<AnalysisIntParameters[3]<<std::endl;
  std::cout<<"LowADCCut: "<<AnalysisIntParameters[4]<<std::endl;
  return 0;
}

int FFTAnalysisModule::GetArray(std::string ArrayName,
                                std::vector<double> &Output) {
  auto Size = ArraySizes[ArrayName];
  Output.resize(Size);
  auto ptr = HostArrays[ArrayName];
  for (int i = 0; i < Size; i++) {
    Output[i] = ptr[i];
  }
  return 0;
}

int FFTAnalysisModule::GetCaloArray(std::string ArrayName,
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
