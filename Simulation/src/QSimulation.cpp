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

  return 0;
}
