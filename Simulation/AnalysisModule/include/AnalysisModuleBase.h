#ifndef ANALYSIS_MODULE_BASE_H
#define ANALYSIS_MODULE_BASE_H

#include <vector>
#include <string>
#include <map>

#include "Global.h"


namespace QAnalysis{
  class AnalysisModule{
    public:
      AnalysisModule(std::string Name, const std::map<std::string,int>& tIntParameters,const std::map<std::string,float> & tFloatParameters,const std::map<std::string,std::string> & tStringParameters);
      std::string GetName();
      void DeviceMemoryReset();
      virtual int FlushAnalysis(std::map<std::string,float *> * SimulatorHostArrays,std::map<std::string,float *> * SimulatorDeviceArrays, std::map<std::string,int> * SimulatorArraySizes) = 0;
      virtual int EndAnalysis(std::map<std::string,float *> * SimulatorHostArrays,std::map<std::string,int> * SimulatorArraySizes) = 0;
      virtual int Output(int RunNumber) = 0;
//    protected:
      std::string fName;
      //Parameter Map, arrays and device arrays
      std::map<std::string,int> IntParameters;
      std::map<std::string,float> FloatParameters;
      std::map<std::string,std::string> StringParameters;

      std::vector<float> AnalysisParameters;
      std::vector<int> AnalysisIntParameters;

      int * d_AnalysisIntParameters;
      float * d_AnalysisParameters;

      // data arrays
      std::map<std::string,float *> HostArrays;
      std::map<std::string,float *> DeviceArrays;
      std::map<std::string,int> ArraySizes;

  };
}//end namespace QAnalysis

#endif 
