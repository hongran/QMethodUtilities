import os
import ROOT
import numpy as np
#import matplotlib.pyplot as plt

ROOTModulePath = os.getenv("QROOTDIR")
InputDataPath = os.getenv("QINPUTDATADIR")
OutputPath = os.getenv("QHISTOGRAMDIR")
InputFileName = "MergedData9d_1213.root"
OutputFileName = "QHists9d.root"
RebinFactor  = 2
DataSetName = "9day"

RecompileModules = True
#RecompileModules = False

#Build the histograms
if RecompileModules:
  HistBuilderModule = ROOTModulePath + "/" + "QHistBuilder.C++"
  QFitModule = ROOTModulePath + "/" + "QFit.C++"
else:
  HistBuilderModule = ROOTModulePath + "/" + "QHistBuilder_C.so"
  QFitModule = ROOTModulePath + "/" + "QFit_C.so"

#cmd = "root -q -b -l '{0}(\"{1}\",\"{2}\",\"{3}\",\"{4}\")'".format(HistBuilderModule,InputFileName,InputDataPath,OutputFileName,OutputPath)
#os.system(cmd)
ROOT.gROOT.ProcessLine(".L {0}".format(HistBuilderModule))
from ROOT import QHistBuilder

ROOT.gROOT.ProcessLine(".L {0}".format(QFitModule))
from ROOT import QFit

QHistBuilder(RebinFactor,InputFileName,InputDataPath,OutputFileName,OutputPath)

HistFile = ROOT.TFile(OutputPath+"/"+OutputFileName,"read")

QHistTotal = HistFile.Get("hCaloHistTotal")
QHistSmoothTotal = HistFile.Get("hCaloHistSmoothTotal")
QHistCalos = []
QHistCaloSmooths = []
QVecCalos = []
QVecCaloTimes = []

for i in range(24):
  QHistCalos.append(HistFile.Get("hCalo_{:02d}".format(i)))
  QHistCaloSmooths.append(HistFile.Get("hCaloSmooth_{:02d}".format(i)))
  QVecCalos.append(HistFile.Get("vCalo_{:02d}".format(i)))
  QVecCaloTimes.append(HistFile.Get("vCaloTime_{:02d}".format(i)))

c1 = ROOT.TCanvas("c1","c1",0,0,1200,800)
c1.Divide(6,4)

for i in range(24):
  c1.cd(i+1)
  QHistCalos[i].Draw()
  QHistCaloSmooths[i].SetLineColor(ROOT.kRed)
  QHistCaloSmooths[i].Draw("same")

c1.Update()

c3 = ROOT.TCanvas("c3","c3",0,0,800,600)

QHistSmoothTotal.Draw()
c3.Update()

print("Start Fitting...")
QFitter = QFit(DataSetName)

fFit = QFitter.Fit(QHistSmoothTotal,61,260,"f10ParFit")

fFit.Draw("same")

c3.Update()

hRes = QFitter.GetResidual()
hResFFT = QFitter.GetResidualFFT()

c4 = ROOT.TCanvas("c4","c4",0,0,1200,600)
c4.Divide(2,1)
c4.cd(1)
hRes.Draw()
c4.cd(2)
hResFFT.Draw()

#fig = plt.figure()
#plt.plot(QVecCaloTimes[0],QVecCalos[0],'ol')
#plt.show()

c4.Update()

