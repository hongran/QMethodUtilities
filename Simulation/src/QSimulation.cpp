#include <iostream>
#include "QSim.h"
#include <vector>
#include <string>

#include "TH1.h"
#include "TFile.h"
#include "TString.h"
#include "TSpline.h"

std::vector<float> IntegratePulsedTemplate(std::string TemplatePath,int CrystalId,int TemplateSize,int TemplateZero);

int main(int argc, char ** argv)
{
  //Get Input arguments
  int RunNumber = std::stoi(std::string(argv[1]));
  int NFlushes = std::stoi(std::string(argv[2]));
  int NFillsPerBatch = std::stoi(std::string(argv[3]));
  //Get Template
  auto Template = IntegratePulsedTemplate("/home/rhong/QMethodUtilities/Simulation/templates",25,2000,200);

  QSimulation::QSim QSimulator(256,1,500,NFillsPerBatch,-999,4,false,false);
  QSimulator.SetIntegratedPulseTemplate(Template,2000,200);
  QSimulator.Simulate(NFlushes);
  
  std::vector<double> QHist;
  QSimulator.GetCaloArray("fillSumArray",QHist);

  unsigned int N = QHist.size();

  TH1 * h = new TH1D("test","test",N,0,N*0.075);
  for (unsigned int i=0;i<N;i++)
  {
    h->SetBinContent(i,QHist[i]);
    h->SetBinError(i,sqrt(QHist[i]));
  }
  
  TH1 * hTemplate = new TH1D("Template","Template",2000,-20,180);
  for (int i=0;i<2000;i++)
  {
    hTemplate->SetBinContent(i,Template[i]);
  }

  TFile* FileOut = new TFile(Form("TestOut_%04d.root",RunNumber),"recreate");
  h->Write();
  hTemplate->Write();
  FileOut->Close();

  delete FileOut;
  delete h;

  return 0;
}

std::vector<float> IntegratePulsedTemplate(std::string TemplatePath,int CrystalId,int TemplateSize,int TemplateZero )
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

