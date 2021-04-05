#include "AnalysisModuleList.h"
#include "QSim.h"


int QSimulation::QSim::RegisterAnalysisModule(std::string ModuleName, const std::map<std::string,int>& tIntParameters,const std::map<std::string,float> & tFloatParameters,const std::map<std::string,std::string> & tStringParameters, int nFlushesPerBatch, int FillMaxLength)
{
  QAnalysis::AnalysisModule * AnalyzerHandle;
  if (ModuleName.compare("RHAnalysis")==0)
  {
    AnalyzerHandle = new QAnalysis::RHAnalysisModule(ModuleName,tIntParameters,tFloatParameters,tStringParameters,nFlushesPerBatch, FillMaxLength);
  }
  if (ModuleName.compare("RPAnalysis")==0)
  {
    AnalyzerHandle = new QAnalysis::RPAnalysisModule(ModuleName,tIntParameters,tFloatParameters,tStringParameters,nFlushesPerBatch, FillMaxLength);
  }
  if (ModuleName.compare("FFTAnalysis")==0)
  {
    AnalyzerHandle = new QAnalysis::FFTAnalysisModule(ModuleName,tIntParameters,tFloatParameters,tStringParameters,nFlushesPerBatch, FillMaxLength);
  }
  
  if (ModuleName.compare("TruthAnalysis")==0)
  {
    AnalyzerHandle = new QAnalysis::TruthAnalysisModule(ModuleName,tIntParameters,tFloatParameters,tStringParameters,nFlushesPerBatch, FillMaxLength);
  }

  if (ModuleName.compare("EnergyHistogram")==0)
  {
    AnalyzerHandle = new QAnalysis::EnergyHistogramModule(ModuleName,tIntParameters,tFloatParameters,tStringParameters,nFlushesPerBatch, FillMaxLength);
  }
  
  AnaModules[ModuleName] = AnalyzerHandle;
  return 0;
}

