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

class QFit{
  public:
    QFit();
    ~QFit();
    //Access Functions
    TF1 GetFunction(std::string fName);
    TF1 GetFitFunction();
    TH1D GetResidual();
    TF1 Fit(TH1 *hist,double FitStart,double FitEnd,std::string fName=std::string("f5ParFit"));
  private:
    //Fit Functions
    std::map<std::string,TF1> fFunctionMap_;
    std::string fFitName_;
    //Residual
    TH1D hResidual_;
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

TH1D QFit::GetResidual()
{
  return hResidual_;
}

TF1 QFit::Fit(TH1 *hist,double FitStart,double FitEnd,std::string fName)
{
  fFitName_ = fName;
  hist->Fit(&fFunctionMap_[fFitName_],"N0E","",FitStart,FitEnd);
  return fFunctionMap_[fFitName_];
}

