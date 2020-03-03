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
  QFitModule = ROOTModulePath + "/" + "QFit.C++"
else:
  QFitModule = ROOTModulePath + "/" + "QFit_C.so"

ROOT.gROOT.ProcessLine(".L {0}".format(QFitModule))
from ROOT import QFit

QFitter = QFit(DataSetName)

#Fake Field Distribution
myField = ROOT.TH1D("myField","myField",7,-35,35)
myField.SetBinContent(1,0.0)
myField.SetBinContent(2,0.1)
myField.SetBinContent(4,0.3)
myField.SetBinContent(5,0.3)
myField.SetBinContent(5,0.2)
myField.SetBinContent(7,0.1)

QFitter.SetFieldDistribution(myField)

hTest = QFitter.GetHistogram("f5ParFit","hTest",1000000000,2000,0,300)
hField = QFitter.GetFieldDistribution()
AvgField = hField.GetMean()

c1 = ROOT.TCanvas("c1","c1",0,0,1200,600)
c1.Divide(2,1)
c1.cd(1)
hTest.Draw()

fFit = QFitter.Fit(hTest,61,260,"f5ParFit")

RShift = (fFit.GetParameter(0)-1.44)/1.44*1e6

fFit.Draw("same")

c1.cd(2)
hField.Draw()

c1.Update()

hRes = QFitter.GetResidual()
hResFFT = QFitter.GetResidualFFT()

c2 = ROOT.TCanvas("c4","c4",0,0,1200,600)
c2.Divide(2,1)
c2.cd(1)
hRes.Draw("hist p")
c2.cd(2)
hResFFT.Draw()

#fig = plt.figure()
#plt.plot(QVecCaloTimes[0],QVecCalos[0],'ol')
#plt.show()

c2.Update()

print("Average Field ",AvgField)
print("R Shift ",RShift)

