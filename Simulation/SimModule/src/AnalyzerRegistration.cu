#include "AnalysisModuleList.h"
#include "QSim.h"


int QSimulation::QSim::RegisterAnalysisModule(std::string ModuleName, const std::map<std::string,int>& tIntParameters,const std::map<std::string,float> & tFloatParameters,const std::map<std::string,std::string> & tStringParameters, int nFlushesPerBatch, int FillMaxLength)
{
  QAnalysis::AnalysisModule * AnalyzerHandle;
  if (ModuleName.compare("RHAnalysis")==0)
  {
    AnalyzerHandle = new QAnalysis::RHAnalysisModule(ModuleName,tIntParameters,tFloatParameters,tStringParameters,nFlushesPerBatch, FillMaxLength);
  }
  AnaModules[ModuleName] = AnalyzerHandle;
  return 0;
}

