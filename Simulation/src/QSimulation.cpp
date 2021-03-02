#include <iostream>
#include <vector>
#include <string>

#include "QSim.h"
#include "AnalysisModuleList.h"
#include "Util.h"
#include "Global.h"

#include "TH1.h"
#include "TFile.h"
#include "TString.h"
#include "TSpline.h"

int Debug = 0;
int Verbose = 0;

int main(int argc, char ** argv)
{
  //parsing the input arguments
  std::vector<std::string> ArgList;
  for (int i=1;i<argc;i++){
    ArgList.push_back(std::string(argv[i]));
  }

  //Get Input arguments
  int RunNumber = 0;
  int NFlushes = 1;
  long long int Seed = 0;
  std::string FileName = ""; //Config File Name

  auto NArg = ArgList.size();
  for (int i = 0; i < NArg - 1; i++)
  {
    if (ArgList[i][0] == '-')
    {
      if (ArgList[i].compare("-c") == 0)
      {
        FileName = ArgList[i + 1];
      }
      else if (ArgList[i].compare("-run") == 0)
      {
        RunNumber = std::stoi(ArgList[i + 1]);
      }
      else if (ArgList[i].compare("-flush") == 0)
      {
        NFlushes = std::stoi(ArgList[i + 1]);
      }
      else if (ArgList[i].compare("-seed") == 0)
      {
        Seed = std::stoll(ArgList[i + 1]);
      }
      else if (ArgList[i].compare("-debug") == 0)
      {
        Debug = std::stoi(ArgList[i + 1]);
      }
      else if (ArgList[i].compare("-verbose") == 0)
      {
        Verbose = std::stoi(ArgList[i + 1]);
      }
    }
    else
    {
      continue;
    }
  }
  if (FileName.compare("") == 0)
  {
    std::cout << "Configuration file is not set." << std::endl;
    return -1;
  }

  json MasterConfig;
  int res = ImportJson(FileName,MasterConfig);
  if (res==-1)
  {
    std::cout << "Configuration file opening failed" <<std::endl;
    return -1;
  }
  std::cout << "Processing configuration file "<<FileName<<std::endl;

  std::map<std::string,int> IntParameters;
  std::map<std::string,float> FloatParameters;
  std::map<std::string,std::string> StringParameters;

  //Config Simulator
  auto SimulatorConfig = GetStructFromJson(MasterConfig,"Simulator");
  JsonToStructs(SimulatorConfig, IntParameters, FloatParameters, StringParameters);

  QSimulation::QSim QSimulator(IntParameters,FloatParameters,StringParameters,Seed);

  //Register Analyzer config
  //std::vector<Simulation::Analyzer> AnalyzerList;
  auto AnalyzerConfigs = GetStructFromJson(MasterConfig,"Analyzers"); 
  for (json::iterator it=AnalyzerConfigs.begin();it!=AnalyzerConfigs.end();++it){
    bool enable = (*it)["Enable"];
    if (!enable)
    {
      continue;
    }
    std::cout <<(*it)["Name"]<<std::endl;
    std::map<std::string,int> AnaIntParameters;
    std::map<std::string,float> AnaFloatParameters;
    std::map<std::string,std::string> AnaStringParameters;
    std::string Name = (*it)["Name"];
    JsonToStructs(*it, AnaIntParameters, AnaFloatParameters, AnaStringParameters);

    QSimulator.RegisterAnalysisModule(Name,AnaIntParameters,AnaFloatParameters,AnaStringParameters,IntParameters["NFlushesPerBatch"],nsPerFill/qBinSize);
  }
  
  QSimulator.Simulate(NFlushes);
  
  std::cout <<"Starting output"<<std::endl;

  QSimulator.Output(RunNumber);

  return 0;
}


