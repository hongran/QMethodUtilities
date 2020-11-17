#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TH1.h"

#include "RPAnalysisModule.h"

// Analysis device kernel
__global__ void flush_analysis(float *FlushQArray, float *AnaQArray,
                               float *AnaPedArray, int *AnalysisIntParameters,
                               float *AnalysisParameters) {
  int flush_buffer_max_length = AnalysisIntParameters[1];
  int nFlushesPerBatch = AnalysisIntParameters[0];
  int wndw = AnalysisIntParameters[2];
  int gap = AnalysisIntParameters[3];

  float threshold = AnalysisParameters[0];

  // thread index
  int iflush = blockIdx.x * blockDim.x + threadIdx.x;

  if (iflush < nFlushesPerBatch) {
    for (int ix = 0; ix < NXSEG; ix++) {
      for (int iy = 0; iy < NYSEG; iy++) {
        int flushoffset = iflush * NSEG * flush_buffer_max_length;
        int xysegmentoffset = (ix + iy * NXSEG) * flush_buffer_max_length;

        for (int iADC = gap + wndw;
             iADC < flush_buffer_max_length - gap - wndw - 1; iADC++) {
          int idx = flushoffset + xysegmentoffset + iADC;
	  float InputBuffer[32];
	  int BufferStartIdx = iADC - gap - wndw;
	  for (int kADC = iADC - gap - wndw; kADC < iADC + gap + wndw + 1;kADC++)
	  {
	    InputBuffer[kADC-BufferStartIdx] = FlushQArray[kADC - iADC + idx];
	  }

          float ysum = 0, yavg = 0;
          // find the mask base on rejection logic: yi - sum(y k!=i)/5 >
          // threshold
          int mask[8] = {1, 1, 1, 1, 1, 1, 1, 1};
          // For samples in the window region left to the trigger sample
          for (int jADC = 0; jADC < wndw; jADC++) {
            ysum = 0;
            for (int kADC = iADC - gap - wndw; kADC < iADC + gap + wndw + 1;
                 kADC++) {
              if (kADC != jADC + iADC - wndw - gap) {
                if (kADC - iADC + gap < 0)
                  //ysum += FlushQArray[kADC - iADC + idx];
                  ysum += InputBuffer[kADC-BufferStartIdx];

                if (kADC - iADC - gap > 0)
                  //ysum += FlushQArray[kADC - iADC + idx];
                  ysum += InputBuffer[kADC-BufferStartIdx];
              }
            }

            yavg = ysum / (2.0 * wndw - 1);

            // reject the sample above threshold in the pedestal region.
	    /*
            if (FlushQArray[jADC + idx - wndw - gap] - yavg > threshold) {
              mask[jADC] = 0;
            }
	    */
            if (InputBuffer[jADC] - yavg > threshold) {
              mask[jADC] = 0;
            }
          } // End of samples in the window region left to the trigger sample.

          // For samples in the window region right to the trigger sample.
          for (int jADC = wndw; jADC < 2 * wndw; jADC++) {
            ysum = 0;
            for (int kADC = iADC - gap - wndw; kADC < iADC + gap + wndw + 1;
                 kADC++) {
              if (kADC != jADC + iADC - wndw + gap + 1) {
                if (kADC - iADC + gap < 0)
                  //ysum += FlushQArray[kADC - iADC + idx];
                  ysum += InputBuffer[kADC-BufferStartIdx];

                if (kADC - iADC - gap > 0)
                  //ysum += FlushQArray[kADC - iADC + idx];
                  ysum += InputBuffer[kADC-BufferStartIdx];
              }
            }

            yavg = ysum / (2.0 * wndw - 1);

            // reject the sample above threshold in the pedestal region.
	    /*
            if (FlushQArray[jADC + idx - wndw + gap + 1] - yavg > threshold) {
              mask[jADC] = 0;
            }
	    */
            if (InputBuffer[jADC + 2*gap + 1] - yavg > threshold) {
              mask[jADC] = 0;
            }
          } // End of samples in the window region right to the trigger sample.

          // compute the pileup corrected pedestal
          ysum = 0;
	  
          for (int jADC = 0; jADC < wndw; jADC++) {
            //ysum += FlushQArray[idx + jADC - gap - wndw] * mask[jADC];
            ysum +=  InputBuffer[jADC] * mask[jADC];
          }
          for (int jADC = wndw; jADC < 2 * wndw; jADC++) {
            //ysum += FlushQArray[idx + jADC + gap - wndw + 1] * mask[jADC];
            ysum += InputBuffer[jADC + 2*gap + 1] * mask[jADC];
          }
	  
          yavg = ysum / (2.0 * wndw - 1);
          //float ydiff = FlushQArray[idx] - yavg;
          float ydiff = InputBuffer[wndw+gap] - yavg;
          AnaPedArray[idx] = yavg;
	  
          if (ydiff > threshold) {
            AnaQArray[idx] += ydiff;
            // // fill the q2dArr
            // int iCol = idx % nBins;
            // int iRow = __double2int_rd(ydiff + 1000) % 420;
            // q2dArr[iRow * nBins + iCol] += 1;
          }
	  
        }
      }
    }
  }
}

namespace QAnalysis {
RPAnalysisModule::RPAnalysisModule(
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
  ArraySizes["AnaPedArray"] =
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
int RPAnalysisModule::FlushAnalysis(
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

  flush_analysis<<<nblocks, IntParameters["NThreadsPerBlock"]>>>(
      (*SimulatorDeviceArrays)["FlushQArray"], DeviceArrays["AnaQArray"],
      DeviceArrays["AnaPedArray"], d_AnalysisIntParameters,
      d_AnalysisParameters);

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

int RPAnalysisModule::EndAnalysis(
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

int RPAnalysisModule::Output(int RunNumber) {
  std::vector<double> AnaQHist;
  std::vector<double> AnaPedHist;
  this->GetCaloArray("AnaQArray", AnaQHist);
  this->GetCaloArray("AnaPedArray", AnaPedHist, false);

  unsigned int N = AnaQHist.size();

  TH1 *hAnaQ = new TH1D("AnaQHist", "AnaQHist", N, 0, N * 0.075);
  for (unsigned int i = 0; i < N; i++) {
    hAnaQ->SetBinContent(i, AnaQHist[i]);
  }

  TH1 *hAnaPed = new TH1D("AnaPedHist", "AnaPedHist", N, 0, N * 0.075);
  for (unsigned int i = 0; i < N; i++) {
    hAnaPed->SetBinContent(i, AnaPedHist[i]);
  }

  TFile *FileOut =
      new TFile(Form("RPRootOut_%04d.root", RunNumber), "recreate");
  hAnaQ->Write();
  hAnaPed->Write();
  FileOut->Close();

  return 0;
}

// Private Functions
int RPAnalysisModule::InitParameters() {
  AnalysisParameters.resize(1);
  AnalysisIntParameters.resize(4);

  AnalysisParameters[0] = FloatParameters["Threshold"];

  AnalysisIntParameters[0] = IntParameters["NFlushesPerBatch"];
  AnalysisIntParameters[1] = IntParameters["FillBufferMaxLength"];
  AnalysisIntParameters[2] = IntParameters["Window"];
  AnalysisIntParameters[3] = IntParameters["Gap"];

  return 0;
}

int RPAnalysisModule::GetArray(std::string ArrayName,
                               std::vector<double> &Output) {
  auto Size = ArraySizes[ArrayName];
  Output.resize(Size);
  auto ptr = HostArrays[ArrayName];
  for (int i = 0; i < Size; i++) {
    Output[i] = ptr[i];
  }
  return 0;
}

int RPAnalysisModule::GetCaloArray(std::string ArrayName,
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
