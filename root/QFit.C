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


//Function classes
class  FitFunction10Par {
  private:
    double fAcbo;
    double fBcbo;
    double fTauAcbo;
    double fTauBcbo;
    double ft0;
 public:
    FitFunction10Par(){}
    FitFunction10Par(double t0,double Acbo,double Bcbo, double TauAcbo, double TauBcbo):ft0(t0),fAcbo(Acbo),fBcbo(Bcbo),fTauAcbo(TauAcbo),fTauBcbo(TauBcbo)
    {}
   
    double operator() (double *x, double *par) {
      double t = x[0];

      //Basic function
      /*
      par[0]: Normalization
      par[1]: phase shift
      par[2]: asymmetry 
      par[3]: lifetiem
      par[4]: normalization
      par[5]: constant background
      */
      double f = par[4]*exp(-(t-ft0)/par[3])*(1+par[2]*cos(par[0]*(t-ft0)+par[1]))+par[5];

      //par[6]: a_cbo;
      //par[7]: tau_cbo;
      //par[8]: frequency constant for cbo frequency
      //par[9]: phi_cbo
      double omega_cbo =  par[8]*(1.0 + fAcbo*exp(-(t-ft0)/fTauAcbo)/(par[8]*(t - ft0)) + fBcbo*exp(-(t-ft0)/fTauBcbo)/ (par[8]*(t-ft0)));

      double n_cbo = 1.0 - par[6]* exp( -( t - ft0 ) / par[7] ) * cos( omega_cbo * ( t - ft0 )   +  par[9] ) ;

      f *= n_cbo;
      return f;

   }
};

Double_t fprec(Double_t *x, Double_t *par);

class QFit{
  public:
    QFit(std::string DataSetName);
    ~QFit();
    //Access Functions
    TF1 GetFunction(std::string fName);
    TF1 GetFitFunction();
    TH1* GetResidual();
    TH1* GetResidualFFT();
    TH1* GetFieldDistribution();
    TH1* GetHistogram(std::string Function,std::string HistName,int Norm, int NBins, double TStart,double TEnd);

    //Set Functions
    int SetFieldDistribution(TH1* fieldDist);
    //Fit Function
    TF1 Fit(TH1 *hist,double FitStart,double FitEnd,std::string fName=std::string("f5ParFit"));
    
  private:
    //data set name
    std::string fDataSet;
    //Fit Functions
    FitFunction10Par f10Par;

    std::map<std::string,TF1> fFunctionMap_;
    std::string fFitName_;
    //Residual
    TH1* hResidual_;
    TH1* hResFFT_;
    //Field
    TH1* hFieldDistribution_;
};

QFit::QFit(std::string DataSetName):fDataSet(DataSetName)
{
  double t0 = 0;
  //Initializing the constants for dataset
  double Acbo = 0.0;
  double Bcbo = 0.0;
  double TauAcbo = 0.0;
  double TauBcbo = 0.0;
  if (DataSetName.compare("60hr")==0)
  {
    Acbo = 2.79;
    Bcbo = 5.63;
    TauAcbo = 61.1;
    TauBcbo = 6.07;
  }else if (DataSetName.compare("9day")==0)
  {
    Acbo = 2.8;
    Bcbo = 6.18;
    TauAcbo = 56.6;
    TauBcbo = 6.32;
  }

  //function objects
  f10Par = FitFunction10Par(t0,Acbo,Bcbo,TauAcbo,TauBcbo);

  //Default fit
  fFitName_ = "f5ParFit";
  //Function List
  fFunctionMap_["f5ParFit"] = TF1("f5ParFit","[4]*exp(-x/[3])*(1+[2]*cos([0]*x+[1]))",30,300);
  fFunctionMap_["f5ParFit"].SetNpx(2000);
  fFunctionMap_["f5ParFit"].SetParameters(1.44,-4.7,0.232,64.33,2.45e8);

  fFunctionMap_["f10ParFit"] = TF1("f10ParFit",f10Par,30,300,10);
  fFunctionMap_["f10ParFit"].SetNpx(2000);
  fFunctionMap_["f10ParFit"].SetParameters(1.44,-4.7,0.232,64.33,2.45e8,0.0,0.29,143.3,2.33,0.0);
  fFunctionMap_["f10ParFit"].FixParameter(5,0.0);

  //Residual
  hResidual_ = nullptr;
  hResFFT_ = nullptr;
  
  //Default Field Distribution
  hFieldDistribution_ = new TH1D("FieldDistribution","FieldDistribution",1,-1e-9,1e-9);
  hFieldDistribution_->SetBinContent(1,1);
}

QFit::~QFit()
{
  //Let root handle this by itself
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

TH1* QFit::GetFieldDistribution()
{
  return (TH1*)hFieldDistribution_->Clone();
}

TH1* QFit::GetHistogram(std::string Function,std::string HistName,int Norm, int NBins ,double TStart,double TEnd)
{
  if (fFunctionMap_.find(Function) == fFunctionMap_.end())
  {
    std::cout << "Function " << Function <<" is not found."<<std::endl;
  }
  auto func = fFunctionMap_[Function];
  
  TH1* Hist = new TH1D(HistName.c_str(),HistName.c_str(),NBins,TStart,TEnd);

  int NField = hFieldDistribution_->GetNbinsX();

  double omega_0 = func.GetParameter(0);

  for (int j=1;j<=NField;j++)
  {
    double shift = hFieldDistribution_->GetBinCenter(j)*1e-6;
    double weight = hFieldDistribution_->GetBinContent(j);
    std::cout << "shift "<<shift<<" ; weight "<<weight<<std::endl;
    func.SetParameter(0,omega_0*(1.0+shift));
    func.SetParameter(4,Norm*weight);

    for (int i=1;i<=NBins;i++)
    {
      double x = Hist->GetBinCenter(i);
      double y = func.Eval(x);
      double y_new = Hist->GetBinContent(i) + y;
      Hist->SetBinContent(i,y_new);
      Hist->SetBinError(i,sqrt(y_new));
    }
  }

  return Hist;
}

int QFit::SetFieldDistribution(TH1* fieldDist)
{
  if (hFieldDistribution_ != nullptr)
  {
    delete hFieldDistribution_;
  }
  hFieldDistribution_ = (TH1*)fieldDist->Clone();
  return 0;
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


