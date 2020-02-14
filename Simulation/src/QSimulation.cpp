#include <iostream>
#include "QSim.h"
#include <vector>
#include <string>

#include "TH1.h"
#include "TFile.h"

int main()
{
  QSimulation::QSim QSimulator(256,1,500,1024,-999,4,false,false);
  QSimulator.Simulate(2048);
  
  std::vector<double> QHist;
  QSimulator.GetCaloArray("fillSumArray",QHist);

  unsigned int N = QHist.size();

  TH1 * h = new TH1D("test","test",N,0,N);
  for (unsigned int i=0;i<N;i++)
  {
    h->SetBinContent(i,QHist[i]);
  }

  TFile* FileOut = new TFile("TestOut.root","recreate");
  h->Write();
  FileOut->Close();

  delete FileOut;
  delete h;

  return 0;
}
