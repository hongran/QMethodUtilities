#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TH1.h"

#include "QMethodStructs.hh"

void QHistBuilder(std::string InputFile,std::string InputDataPath,std::string OutputFile,std::string OutputDataPath)
{
  double clk = 1.25e-3;//us
  double TFlashVeto = 35.0;//us

  double BinSize = clk*60;

//  int SmoothWidth = 1;

  std::string InputFileName = InputDataPath+std::string("/")+InputFile;
  std::cout  << "Opening the input root file " << InputFileName <<std::endl;

  std::vector<gm2analyses::QHistCaloSum_t> QHists(24);
  std::vector<gm2analyses::QHistCaloSum_t> QHistsSum(24);

  TFile * DataIn = new TFile(InputFileName.c_str(),"read");

  TTree * TreeInput = (TTree*)DataIn->Get("QTree/tQMethodJob");

  for (int i=0;i<24;i++){
    TreeInput->SetBranchAddress(Form("QHistCalo_%02d",i),&QHists[i]);
  }

  auto NEntries = TreeInput->GetEntries();
  for (int j=0;j<NEntries;j++)
  {
    TreeInput->GetEntry(j);
    for (int i=0;i<24;i++){
      for (int k=0;k<NQBINSOUT;k++)
      {
	QHistsSum[i].Signal[k] += QHists[i].Signal[k];
	QHistsSum[i].SignalErr[k] = sqrt(pow(QHists[i].SignalErr[k],2.0)+pow(QHists[i].SignalErr[k],2.0));
      }
    }
  }

  DataIn->Close();
  delete DataIn;

  //Fill the histograms
  TH1 * CaloHist[24];
  TH1 * CaloHistSmooth[24]; //Filter out the fast rotation

  double TimeStart = OFFSET*clk;
  double TimeEnd = (NQBINSOUT*60+OFFSET)*clk;
  std::vector<double> TimeVec(NQBINSOUT);
  for (int k=1;k<NQBINSOUT;k++)
  {
    TimeVec[k] = TimeStart + k*BinSize - BinSize/2.0;
  }

  for (int i=0;i<24;i++)
  {
    CaloHist[i] = new TH1D(Form("hCalo_%02d",i),Form(";Time [#mus];Counts"),NQBINSOUT,TimeStart,TimeEnd);
    CaloHistSmooth[i] = new TH1D(Form("hCaloSmooth_%02d",i),Form(";Time [#mus];Counts"),NQBINSOUT,TimeStart,TimeEnd);
    for (int k=1;k<NQBINSOUT;k++)
    {
      //remove flash
      if (TimeStart+k*BinSize-BinSize/2.0<TFlashVeto){
	continue;
      }
      CaloHist[i]->SetBinContent(k,QHistsSum[i].Signal[k]);
      CaloHist[i]->SetBinError(k,QHistsSum[i].SignalErr[k]);
      if (k>1 && k<NQBINSOUT-1)
      {
	double val = QHistsSum[i].Signal[k]/2.0+QHistsSum[i].Signal[k-1]/4.0+QHistsSum[i].Signal[k+1]/4.0;
	double Err2 = pow(QHistsSum[i].SignalErr[k]/2.0,2.0)+pow(QHistsSum[i].SignalErr[k-1]/4.0,2.0)+pow(QHistsSum[i].SignalErr[k+1]/4.0,2.0);
	CaloHistSmooth[i]->SetBinContent(k,val);
	CaloHistSmooth[i]->SetBinError(k,sqrt(Err2));
      }
    }
  }

  //Sum over calos
  TH1 * CaloHistTotal = (TH1*)CaloHist[0]->Clone();;
  TH1 * CaloHistSmoothTotal = (TH1*)CaloHistSmooth[0]->Clone();;
  CaloHistTotal->SetName("hCaloHistTotal");
  CaloHistSmoothTotal->SetName("hCaloHistSmoothTotal");
  for (unsigned int icalo = 1;icalo<24;icalo++){
    CaloHistTotal->Add(CaloHist[icalo]);
    CaloHistSmoothTotal->Add(CaloHistSmooth[icalo]);
  }

  //Output
  std::string OutputFileName = OutputDataPath+std::string("/")+OutputFile;
  TFile * FileOut = new TFile(OutputFileName.c_str(),"recreate");
  for (unsigned int icalo = 0;icalo<24;icalo++){
    CaloHist[icalo]->Write();
    CaloHistSmooth[icalo]->Write();
    //Write std vectors
    std::vector<double> QVals(QHistsSum[icalo].Signal,QHistsSum[icalo].Signal+NQBINSOUT);
    std::vector<double> QErrors(QHistsSum[icalo].SignalErr,QHistsSum[icalo].SignalErr+NQBINSOUT);
    FileOut->WriteObject(&QVals,Form("vCalo_%02d",icalo));
    FileOut->WriteObject(&QErrors,Form("vCaloErr_%02d",icalo));
    FileOut->WriteObject(&TimeVec,Form("vCaloTime_%02d",icalo));
  }
  CaloHistTotal->Write();
  CaloHistSmoothTotal->Write();

  FileOut->Close();
  delete FileOut;
}

