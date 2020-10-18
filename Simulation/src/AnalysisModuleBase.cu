#include <vector>
#include <string>

#include "AnalysisModuleBase.h"


namespace QAnalysis{
  AnalysisModule::AnalysisModule(std::string Name, const std::map<std::string,int>& tIntParameters,const std::map<std::string,float> & tFloatParameters,const std::map<std::string,std::string> & tStringParameters)
  {
    //Import Parameters
    IntParameters = tIntParameters;
    FloatParameters = tFloatParameters;
    StringParameters = tStringParameters;
  }

  std::string AnalysisModule::GetName()
  {
    return fName;
  }

  void AnalysisModule::DeviceMemoryReset()
  {
    for (auto it=ArraySizes.begin();it!=ArraySizes.end();++it)
    {
      auto Name = it->first;
      auto Size = it->second;
      cudaMemset( DeviceArrays[Name], 0.0, Size);
    }
  }

}//end namespace QAnalysis

