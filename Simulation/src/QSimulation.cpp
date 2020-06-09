#include <iostream>
#include <vector>
#include <string>

#include "QSim.h"
#include "Util.h"
#include "Global.h"

#include "TH1.h"
#include "TFile.h"
#include "TString.h"
#include "TSpline.h"

std::vector<float> IntegratePulseTemplate(std::string TemplatePath,int CrystalId,int TemplateSize,int TemplateZero);
std::vector<float> LoadPedestalTemplate(std::string TemplatePath,int CrystalId);
int JsonToStructs(const json & Config,std::map<std::string,float>& FloatParameters, std::map<std::string,int>& IntParameters);

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
    ;
  }

  json MasterConfig;
  int res = ImportJson(FileName,MasterConfig);
  if (res==-1)
  {
    std::cout << "Configuration file opening failed" <<std::endl;
    return -1;
  }
  std::cout << "Processing configuration file "<<FileName<<std::endl;

  std::string PulseTemplatePath = GetValueFromJson<std::string>(MasterConfig,"Pulse Template Path");
  std::string PedTemplatePath = GetValueFromJson<std::string>(MasterConfig,"Pedestal Template Path");
  json SimulatorConfig = GetStructFromJson(MasterConfig,"Simulator");

  std::map<std::string,float> FloatParameters;
  std::map<std::string,int> IntParameters;
  JsonToStructs(SimulatorConfig, FloatParameters, IntParameters);

  //int NFillsPerBatch = std::stoi(std::string(argv[3]));
  //Get Template
  auto PulseTemplate = IntegratePulseTemplate(PulseTemplatePath,25,2000,200);
  auto PedestalTemplate = LoadPedestalTemplate(PedTemplatePath,0);

  QSimulation::QSim QSimulator(FloatParameters,IntParameters,Seed);
  QSimulator.SetIntegratedPulseTemplate(PulseTemplate,2000,200);
  QSimulator.SetPedestalTemplate(PedestalTemplate);
  QSimulator.Simulate(NFlushes);
  
  std::cout <<"Starting output"<<std::endl;

  std::vector<double> FillSumHist;
  std::vector<double> LastFillHist;
  std::vector<double> QHist;
  QSimulator.GetCaloArray("fillSumArray",FillSumHist);
  QSimulator.GetCaloArray("fillSumArrayPed",LastFillHist,false);
  QSimulator.GetCaloArray("analyzedQSumArray",QHist);

  unsigned int N = QHist.size();

  TH1 * hSum = new TH1D("SumHist","SumHist",N,0,N*0.075);
  for (unsigned int i=0;i<N;i++)
  {
    if (i>3100)break;
    hSum->SetBinContent(i,FillSumHist[i]);
    //h->SetBinError(i,sqrt(QHist[i]));
  }
  
  TH1 * hLastFill = new TH1D("LastFillHist","LastFillHist",N,0,N*0.075);
  for (unsigned int i=0;i<N;i++)
  {
    if (i>3100)break;
    hLastFill->SetBinContent(i,LastFillHist[i]);
    //h->SetBinError(i,sqrt(QHist[i]));
  }
  
  TH1 * hQ = new TH1D("QHist","QHist",N,0,N*0.075);
  for (unsigned int i=107;i<N;i++)
  {
    if (i>3100)break;
    hQ->SetBinContent(i,QHist[i]);
    //h->SetBinError(i,sqrt(QHist[i]));
  }
  
  TH1 * hTemplate = new TH1D("Template","Template",2000,-20,180);
  for (int i=0;i<2000;i++)
  {
    hTemplate->SetBinContent(i,PulseTemplate[i]);
  }

  TH1 * hPedTemplate = new TH1D("PedTemplate","PedTemplate",PedestalTemplate.size(),0,PedestalTemplate.size());
  for (int i=0;i<PedestalTemplate.size();i++)
  {
    hPedTemplate->SetBinContent(i,PedestalTemplate[i]);
  }

  TFile* FileOut = new TFile(Form("TestOut_%04d.root",RunNumber),"recreate");
  hSum->Write();
  hLastFill->Write();
  hQ->Write();
  hTemplate->Write();
  hPedTemplate->Write();
  FileOut->Close();

  delete FileOut;
  delete hSum;
  delete hLastFill;
  delete hQ;

  return 0;
}

std::vector<float> IntegratePulseTemplate(std::string TemplatePath,int CrystalId,int TemplateSize,int TemplateZero )
{
  std::string FileName = TemplatePath + "/template" + std::to_string(CrystalId) + ".root";
  TFile * TemplateFile = new TFile(FileName.c_str(),"read");
  auto TemplateSpline = (TSpline3*)TemplateFile->Get("masterSpline");

  std::vector<float> Template(TemplateSize);

  float AccumulatedVal = 0.0;
  for (int i=0;i<TemplateSize;i++)
  {
    float t = static_cast<float>(i - TemplateZero)/10.0;
    float val = TemplateSpline->Eval(t);
    AccumulatedVal += val;
    Template[i] = AccumulatedVal;
  }
  float norm = Template[TemplateSize-1];
  for (int i=0;i<TemplateSize;i++)
  {
    Template[i] /= norm;
  }

  TemplateFile->Close();
  delete TemplateFile;

  return Template;
}

std::vector<float> LoadPedestalTemplate(std::string TemplatePath,int CrystalId)
{
  //std::string FileName = TemplatePath + "/template" + std::to_string(CrystalId) + ".root";
  std::string FileName = TemplatePath + "/PedTemplate" + ".root";
  TFile * TemplateFile = new TFile(FileName.c_str(),"read");
  auto PedTemplate = (TH1*)TemplateFile->Get("hCalo_Pedestal_10");

  unsigned int Size = PedTemplate->GetNbinsX();

  std::vector<float> Template(Size);
  double norm = 2e-4;

  for (int i=0;i<Size;i++)
  {
    Template[i] = PedTemplate->GetBinContent(i) * norm;
  }
  TemplateFile->Close();
  delete TemplateFile;

  return Template;
}

int JsonToStructs(const json & Config, std::map<std::string,float>& FloatParameters, std::map<std::string,int>& IntParameters)
{
  json SimPar = GetStructFromJson(Config,"Simulation Parameters");
  json AnaPar = GetStructFromJson(Config,"Analysis Parameters");

  FloatParameters["Omega_a"] = GetValueFromJson<float>(SimPar,"Omega_a");
  FloatParameters["Noise"] = GetValueFromJson<float>(SimPar,"Noise");

  IntParameters["NThreadsPerBlock"] = GetValueFromJson<int>(SimPar,"NThreadsPerBlock");
  IntParameters["NElectronsPerFill"] = GetValueFromJson<int>(SimPar,"NElectronsPerFill");
  IntParameters["NFillsPerFlush"] = GetValueFromJson<int>(SimPar,"NFillsPerFlush");
  IntParameters["NFlushesPerBatch"] = GetValueFromJson<int>(SimPar,"NFlushesPerBatch");
  IntParameters["TemplateSize"] = GetValueFromJson<int>(SimPar,"TemplateSize");
  IntParameters["TemplateZero"] = GetValueFromJson<int>(SimPar,"TemplateZero");
  IntParameters["FillNoiseSwitch"] = GetValueFromJson<int>(SimPar,"FillNoiseSwitch");
  IntParameters["FlashGainSagSwitch"] = GetValueFromJson<int>(SimPar,"FlashGainSagSwitch");

  FloatParameters["Threshold"] = GetValueFromJson<float>(AnaPar,"Threshold");

  IntParameters["AnalysisSwitch"] = GetValueFromJson<float>(AnaPar,"AnalysisSwitch");
  IntParameters["Window"] = GetValueFromJson<int>(AnaPar,"Window");
  IntParameters["Gap"] = GetValueFromJson<int>(AnaPar,"Gap");
}

