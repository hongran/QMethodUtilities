#include <iostream>
#include "QSim.h"
#include <vector>
#include <string>

#include "TH1.h"
#include "TFile.h"

int main()
{
  QSimulation::QSim QSimulator(128,256,500,127,-999,4,false,false);
  QSimulator.Simulate(1024);
  
  std::vector<double> QHist;

  return 0;
}
