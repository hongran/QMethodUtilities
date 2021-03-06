#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TH1.h"
#include "TVirtualFFT.h"

#include "QMethodStructs.hh"

std::vector<std::vector<TH1*>> QJobDataAccess(int CaloId,std::string InputFile,std::string InputDataPath,std::string OutputFile,std::string OutputDataPath)
{
  double clk = 1.25e-3;//us
  double TFlashVeto = 35.0;//us

  double BinSize = clk*60;

  std::string InputFileName = InputDataPath+std::string("/")+InputFile;
  std::cout  << "Opening the input root file " << InputFileName <<std::endl;

  std::vector<gm2analyses::QHistCrystalSum_t> CrystalData(NCRYSTAL);
  gm2analyses::QHistCaloSum_t CaloData;

  TFile * DataIn = new TFile(InputFileName.c_str(),"read");

  TTree * TreeInput = (TTree*)DataIn->Get(Form("QTree/tQMethodJob"));

  TreeInput->SetBranchAddress(Form("QHistCalo_%02d",CaloId),&CaloData);
  for (int i=0;i<NCRYSTAL;i++)
  {
    TreeInput->SetBranchAddress(Form("QHistCrystal_%02d_%02d",CaloId,i),&CrystalData[i]);
  }

  auto NEntries = TreeInput->GetEntries();

  TreeInput->GetEntry(0);

  DataIn->Close();
  delete DataIn;

  //Return Vectors
  std::vector<std::vector<TH1*>> OutputHists(5);

  //Fill the histograms
  double TimeStart = OFFSET*clk;
  double TimeEnd = (NQBINSOUT*60+OFFSET)*clk;

  //CaloHists;
  TH1 * CaloHist_Raw = new TH1D(Form("hCalo_Raw_%02d",CaloId),Form(";Time [#mus];Counts"),NQBINSOUT,TimeStart,TimeEnd); 
  TH1 * CaloHist_Signal = new TH1D(Form("hCalo_Signal_%02d",CaloId),Form(";Time [#mus];Counts"),NQBINSOUT,TimeStart,TimeEnd); 
  TH1 * CaloHist_Pedestal = new TH1D(Form("hCalo_Pedestal_%02d",CaloId),Form(";Time [#mus];Counts"),NQBINSOUT,TimeStart,TimeEnd); 
  //Calo pedestal FFT
  int n_size = 2000;
  TVirtualFFT *myFFT = TVirtualFFT::FFT(1, &n_size, "R2C EX K");

  for (int k=1;k<NQBINSOUT;k++)
  {
    //remove flash
    if (TimeStart+k*BinSize-BinSize/2.0<TFlashVeto){
      continue;
    }
    CaloHist_Raw->SetBinContent(k,CaloData.Raw[k]);
    CaloHist_Signal->SetBinContent(k,CaloData.Signal[k]);
    CaloHist_Pedestal->SetBinContent(k,CaloData.Pedestal[k]);
  }
  myFFT->SetPoints(&CaloData.Pedestal[400]);
  std::cout <<"Performing FFT"<<std::endl;
  myFFT->Transform();
  std::cout <<"Constructing FFT Histogram"<<std::endl;
  TH1* hPedFFT = nullptr;
  hPedFFT =TH1::TransformHisto(myFFT, hPedFFT, "MAG");

  int NFFT = hPedFFT->GetNbinsX();
  double Range = 2000*60*1.25e-3;
  hPedFFT->SetBins(NFFT,0,NFFT/Range);
  hPedFFT->GetXaxis()->SetRange(0,NFFT/2);
  hPedFFT->GetXaxis()->SetTitle("f [MHz]");

  OutputHists[0].push_back(CaloHist_Raw);
  OutputHists[0].push_back(CaloHist_Pedestal);
  OutputHists[0].push_back(CaloHist_Signal);
  OutputHists[0].push_back(hPedFFT);


  for (int j=0;j<NCRYSTAL;j++)
  {
    //CrystalHists;
    TH1 * CrystalHist_Raw = new TH1D(Form("hCrystal_Raw_%02d_%02d",CaloId,j),Form(";Time [#mus];Counts"),NQBINSOUT,TimeStart,TimeEnd); 
    TH1 * CrystalHist_Signal = new TH1D(Form("hCrystal_Signal_%02d_%02d",CaloId,j),Form(";Time [#mus];Counts"),NQBINSOUT,TimeStart,TimeEnd); 
    TH1 * CrystalHist_Pedestal = new TH1D(Form("hCrystal_Pedestal_%02d_%02d",CaloId,j),Form(";Time [#mus];Counts"),NQBINSOUT,TimeStart,TimeEnd); 

    for (int k=1;k<NQBINSOUT;k++)
    {
      //remove flash
      if (TimeStart+k*BinSize-BinSize/2.0<TFlashVeto){
	continue;
      }
      CrystalHist_Raw->SetBinContent(k,CrystalData[j].Raw[k]);
      CrystalHist_Signal->SetBinContent(k,CrystalData[j].Signal[k]);
      CrystalHist_Pedestal->SetBinContent(k,CrystalData[j].Pedestal[k]);
    }
    OutputHists[1].push_back(CrystalHist_Raw);
    OutputHists[2].push_back(CrystalHist_Pedestal);
    OutputHists[3].push_back(CrystalHist_Signal);
  }

  //Output
  /*
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
  */

  return OutputHists;
}

