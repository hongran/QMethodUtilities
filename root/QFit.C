#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TH1.h"
#include "TF1.h"
#include "TVirtualFFT.h"

class QFit{
  public:
    QFit();
    ~QFit();
    //Access Functions
    TF1 GetFunction(std::string fName);
    TF1 GetFitFunction();
    TH1* GetResidual();
    TH1* GetResidualFFT();
    TF1 Fit(TH1 *hist,double FitStart,double FitEnd,std::string fName=std::string("f5ParFit"));
  private:
    //Fit Functions
    std::map<std::string,TF1> fFunctionMap_;
    std::string fFitName_;
    //Residual
    TH1* hResidual_;
    TH1* hResFFT_;
};

QFit::QFit()
{
  fFitName_ = "f5ParFit";
  fFunctionMap_["f5ParFit"] = TF1("f5ParFit","[4]*exp(-x/[3])*(1+[2]*cos([0]*x+[1]))",0,300);
  fFunctionMap_["f5ParFit"].SetNpx(2000);
  fFunctionMap_["f5ParFit"].SetParameters(1.4,0,0.3,60,9e7);
}

QFit::~QFit()
{
  ;
}


TF1 QFit::GetFunction(std::string fName)
{
  return fFunctionMap_[fName];
}

TF1 QFit::GetFitFunction()
{
  return fFunctionMap_[fFitName_];
}

TH1* QFit::GetResidual()
{
  return (TH1*)hResidual_->Clone();
}

TH1* QFit::GetResidualFFT()
{
  return (TH1*)hResFFT_->Clone();
}

TF1 QFit::Fit(TH1 *hist,double FitStart,double FitEnd,std::string fName)
{
  fFitName_ = fName;
  hist->Fit(&fFunctionMap_[fFitName_],"N0E","",FitStart,FitEnd);
  //Calculate Residual
  int NStart = hist->GetXaxis()->FindBin(FitStart);
  int NStop = hist->GetXaxis()->FindBin(FitEnd);

  hResidual_ = (TH1*)hist->Clone();
  int NBins = hResidual_->GetNbinsX();
  std::vector<double> dIn(NStop-NStart+1);

  for (int i=0;i<NBins;i++)
  {
    if (i<NStart || i>NStop){
      hResidual_->SetBinContent(i,0);
      hResidual_->SetBinError(i,0);
    }else{
      double t = hResidual_->GetBinCenter(i);
      double BinVal = hResidual_->GetBinContent(i);
      double BinErr = hResidual_->GetBinError(i);
      double val = fFunctionMap_[fFitName_].Eval(t);
      hResidual_->SetBinContent(i,BinVal-val);
      hResidual_->SetBinError(i,BinErr);

      dIn[i-NStart] = BinVal-val;
    }
  }
  //FFT of the residual
  int n_size = dIn.size();
  
  std::cout <<"Performing FFT"<<std::endl;
  TVirtualFFT *myFFT = TVirtualFFT::FFT(1, &n_size, "R2C EX K");
  myFFT->SetPoints(&dIn[0]);
  myFFT->Transform();

  std::cout <<"Constructing FFT Histogram"<<std::endl;
  hResFFT_ = nullptr;
  hResFFT_ = TH1::TransformHisto(myFFT, hResFFT_, "MAG");

  int NFFT = hResFFT_->GetNbinsX();
  double Range = hResidual_->GetBinCenter(NStop) - hResidual_->GetBinCenter(NStart);
  hResFFT_->SetBins(NFFT,0,NFFT/Range);
  hResFFT_->GetXaxis()->SetRange(0,NFFT/2);
  hResFFT_->GetXaxis()->SetTitle("f [MHz]");

  return fFunctionMap_[fFitName_];
}

