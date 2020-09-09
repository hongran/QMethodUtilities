#include <vector>
#include <string>
#include <iostream>

#include "TH1.h"
#include "TFile.h"

#include "RHAnalysisModule.h"


//Analysis device kernel
__global__ void flush_analysis(float * FlushHitArray, float * FlushQArray, float * AnaQArray,float * AnaPedArray, int * AnalysisIntParameters, float * AnalysisParameters)
{
  int flush_buffer_max_length = AnalysisIntParameters[1];
  int nFlushesPerBatch = AnalysisIntParameters[0];
  int window = AnalysisIntParameters[2];
  int gap = AnalysisIntParameters[3];

  float threshold = AnalysisParameters[0];

  // thread index
  int iflush = blockIdx.x*blockDim.x + threadIdx.x;

  if (iflush < nFlushesPerBatch) {
    for (int ix = 0; ix < NXSEG; ix++) {
      for (int iy = 0; iy < NYSEG; iy++) {
	int flushoffset = iflush*NSEG*flush_buffer_max_length;
	int xysegmentoffset = (ix+iy*NXSEG)*flush_buffer_max_length;

	for (int i = gap+window; i < flush_buffer_max_length-gap-window-1; i++){ 
	  int index = flushoffset + xysegmentoffset + i;
	  int fill_index = iflush*flush_buffer_max_length + i;

	  float avg = 0;
	  int window_count = 0;
	  for (int k=1; k<=window; k++)
	  {
	    if (FlushHitArray[fill_index - gap - k]<0.5)
	    {
	      avg += FlushQArray[ index - gap - k ];
	      window_count ++;
	    }
	    if (FlushHitArray[fill_index + gap + k]<0.5)
	    {
	      avg += FlushQArray[ index + gap + k ];
	      window_count ++;
	    }
	  }
	  if (window_count == 0)
	  {
	    avg = 0;
	  }else{
	    avg /= (1.0*window_count);
	  }

	  AnaPedArray[index] = avg;

	  if (FlushHitArray[fill_index] > 0)
	  {
	    float signal = FlushQArray[index] - avg;
	    if (signal > threshold)
	    {
	      AnaQArray[index] = signal;
	    }
	  }

	}
      }
    }
  }

}


namespace QAnalysis{
  
  RHAnalysisModule::RHAnalysisModule(std::string Name, const std::map<std::string,int>& tIntParameters,const std::map<std::string,float> & tFloatParameters,const std::map<std::string,std::string> & tStringParameters, int nFlushesPerBatch, int FillMaxLength) : AnalysisModule(Name, tIntParameters, tFloatParameters, tStringParameters)
  {
    //Initialize the parameter arrays
    IntParameters["NFlushesPerBatch"] = nFlushesPerBatch;
    IntParameters["FillBufferMaxLength"] = FillMaxLength;
    InitParameters();

    //Allocate Derive memory for parameters

    cudaMalloc( (void **)&d_AnalysisParameters, AnalysisParameters.size()*sizeof(float));
    cudaMalloc( (void **)&d_AnalysisIntParameters, AnalysisIntParameters.size()*sizeof(int));

    cudaMemcpy( d_AnalysisParameters,&AnalysisParameters[0], AnalysisParameters.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_AnalysisIntParameters,&AnalysisIntParameters[0], AnalysisIntParameters.size()*sizeof(float), cudaMemcpyHostToDevice);

    //Arrays
    ArraySizes["AnaQArray"] = nFlushesPerBatch*NSEG*FillMaxLength*sizeof(float);
    ArraySizes["AnaPedArray"] = nFlushesPerBatch*NSEG*FillMaxLength*sizeof(float);

    //Allocate memories
    for (auto it=ArraySizes.begin();it!=ArraySizes.end();++it)
    {
      auto Name = it->first;
      auto Size = it->second;
      HostArrays[Name] = (float *)malloc(Size);
      cudaMalloc( (void **)&DeviceArrays[Name], Size);
    }
  }

  //Analysis Functions
  int RHAnalysisModule::FlushAnalysis(std::map<std::string,float *> * SimulatorHostArrays,std::map<std::string,float *> * SimulatorDeviceArrays, std::map<std::string,int> * SimulatorArraySizes)
  {
    
    int nblocks = IntParameters["NFlushesPerBatch"] / IntParameters["NThreadsPerBlock"] + 1;
    std::cout << "Analyzing flush batch"<<std::endl;

    flush_analysis<<<nblocks,IntParameters["NThreadsPerBlock"]>>>( (*SimulatorDeviceArrays)["FlushHitArray"], (*SimulatorDeviceArrays)["FlushQArray"], DeviceArrays["AnaQArray"],DeviceArrays["AnaPedArray"], d_AnalysisIntParameters, d_AnalysisParameters);

    cudaDeviceSynchronize();
    auto err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure with user kernel function make)flush anlaysis %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }

    return 0;
  }

  int RHAnalysisModule::EndAnalysis(std::map<std::string,float *> * SimulatorHostArrays, std::map<std::string,int> * SimulatorArraySizes)
  {
    //copy back to host
    int n=0;
    cudaError err;
    for (auto it=ArraySizes.begin();it!=ArraySizes.end();++it)
    {
      auto Name = it->first;
      auto Size = it->second;
      cudaMemcpy( HostArrays[Name], DeviceArrays[Name], Size, cudaMemcpyDeviceToHost);
//      std::cout<< n << " "<<Name<<" "<<Size<<std::endl;
      err=cudaGetLastError();
      if(err!=cudaSuccess) {
        printf("Cuda failure with user kernel function make)analysis endrun copy %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
        exit(0);
      }
      n++;
    }

    return 0;
  }

  int RHAnalysisModule::Output(int RunNumber)
  {
    std::vector<double> AnaQHist;
    std::vector<double> AnaPedHist;
    this->GetCaloArray("AnaQArray",AnaQHist);
    this->GetCaloArray("AnaPedArray",AnaPedHist,false);

    unsigned int N = AnaQHist.size();

    TH1 * hAnaQ = new TH1D("AnaQHist","AnaQHist",N,0,N*0.075);
    for (unsigned int i=0;i<N;i++)
    {
      hAnaQ->SetBinContent(i,AnaQHist[i]);
    }

    TH1 * hAnaPed = new TH1D("AnaPedHist","AnaPedHist",N,0,N*0.075);
    for (unsigned int i=0;i<N;i++)
    {
      hAnaPed->SetBinContent(i,AnaPedHist[i]);
    }

    TFile* FileOut = new TFile(Form("RHRootOut_%04d.root",RunNumber),"recreate");
    hAnaQ->Write();
    hAnaPed->Write();
    FileOut->Close();

    return 0;
  }

  //Private Functions
  int RHAnalysisModule::InitParameters()
  {
    AnalysisParameters.resize(1);
    AnalysisIntParameters.resize(4);

    AnalysisParameters[0] = FloatParameters["Threshold"];

    AnalysisIntParameters[0] = IntParameters["NFlushesPerBatch"];
    AnalysisIntParameters[1] = IntParameters["FillBufferMaxLength"];
    AnalysisIntParameters[2] = IntParameters["Window"];
    AnalysisIntParameters[3] = IntParameters["Gap"];

    return 0;
  }
  
  int RHAnalysisModule::GetArray(std::string ArrayName,std::vector<double>& Output)
  {
    auto Size = ArraySizes[ArrayName];
    Output.resize(Size);
    auto ptr = HostArrays[ArrayName];
    for (int i=0;i<Size;i++)
    {
      Output[i] = ptr[i];
    }
    return 0;
  }

  int RHAnalysisModule::GetCaloArray(std::string ArrayName,std::vector<double>& Output,bool BatchSum)
  {
    Output.clear();
    Output.resize(IntParameters["FillBufferMaxLength"],0.0);
    auto ptr = HostArrays[ArrayName];

    int nFlushesPerBatch = IntParameters["NFlushesPerBatch"];

    for (unsigned int k=0;k<nFlushesPerBatch;k++)
    {
      for (unsigned int j=0;j<NSEG;j++)
      {
        for (unsigned int i=0;i<IntParameters["FillBufferMaxLength"];i++)
        {
          Output[i] += ptr[(k*NSEG+j)*IntParameters["FillBufferMaxLength"] + i];
        }
      }
      if (!BatchSum)
      {
        break;
      }
    }

    return 0;
  }

}//end namespace QAnalysis

