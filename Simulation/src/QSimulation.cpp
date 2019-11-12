#include <iostream>
#include "QSim.h"
#include <vector>
#include <string>

#include "TH1.h"
#include "TFile.h"

int main()
{
  QSimulation::QSim QSimulator(16,1,5500,32,-999,4,false,false);
  QSimulator.Simulate(64);
  
  std::vector<double> QHist;

  return 0;
}
