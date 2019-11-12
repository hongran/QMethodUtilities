import os
import ROOT
import numpy as np
import matplotlib.pyplot as plt

ROOTModulePath = os.getenv("QROOTDIR")
InputDataPath = os.getenv("QINPUTDATADIR")
OutputPath = os.getenv("QHISTOGRAMDIR")
InputFileName = "MergedData9d_1030.root"
OutputFileName = "QHists9d.root"

RecompileModules = True

#Build the histograms
if RecompileModules:
  HistBuilderModule = ROOTModulePath + "/" + "QHistBuilder.C+"
  QFitModule = ROOTModulePath + "/" + "QFit.C+"
else:
  HistBuilderModule = ROOTModulePath + "/" + "QHistBuilder_C.so"
  QFitModule = ROOTModulePath + "/" + "QFit_C.so"

#cmd = "root -q -b -l '{0}(\"{1}\",\"{2}\",\"{3}\",\"{4}\")'".format(HistBuilderModule,InputFileName,InputDataPath,OutputFileName,OutputPath)
#os.system(cmd)
ROOT.gROOT.ProcessLine(".L {0}".format(HistBuilderModule))
from ROOT import QHistBuilder

ROOT.gROOT.ProcessLine(".L {0}".format(QFitModule))
from ROOT import QFit

QHistBuilder(InputFileName,InputDataPath,OutputFileName,OutputPath)

HistFile = ROOT.TFile(OutputPath+"/"+OutputFileName,"read")

QHistTotal = HistFile.Get("hCaloHistTotal")
QHistSmoothTotal = HistFile.Get("hCaloHistSmoothTotal")
QHistCalos = []
QHistCaloSmooths = []
QVecCalos = []
QVecCaloTimes = []

for i in range(24):
  QHistCalos.append(HistFile.Get(f"hCalo_{i:02d}"))
  QHistCaloSmooths.append(HistFile.Get(f"hCaloSmooth_{i:02d}"))
  QVecCalos.append(HistFile.Get(f"vCalo_{i:02d}"))
  QVecCaloTimes.append(HistFile.Get(f"vCaloTime_{i:02d}"))

c1 = ROOT.TCanvas("c1","c1",0,0,1200,800)
c1.Divide(6,4)

for i in range(24):
  c1.cd(i+1)
  QHistCalos[i].Draw()

c1.Update()

c2 = ROOT.TCanvas("c2","c2",0,0,1200,800)
c2.Divide(6,4)

for i in range(24):
  c2.cd(i+1)
  QHistCaloSmooths[i].Draw()

c2.Update()

c3 = ROOT.TCanvas("c3","c3",0,0,800,600)

QHistSmoothTotal.Draw()
c3.Update()

print("Start Fitting...")
QFitter = QFit()

fFit = QFitter.Fit(QHistSmoothTotal,38,260,"f5ParFit")

fFit.Draw("same")

c3.Update()

#fig = plt.figure()
#plt.plot(QVecCaloTimes[0],QVecCalos[0],'ol')
#plt.show()

