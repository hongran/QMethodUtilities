import os
import ROOT
import numpy as np
import matplotlib.pyplot as plt

ROOTModulePath = os.getenv("QROOTDIR")
InputDataPath = os.getenv("QRAWINPUTDATADIR")
OutputPath = os.getenv("QHISTOGRAMDIR")
InputFileName = "gm2offline_qmethTree.root"
OutputFileName = "QJob.root"

RecompileModules = True
#RecompileModules = False

#Build the histograms
if RecompileModules:
  JobDataAccessModule = ROOTModulePath + "/" + "QJobDataAccess.C+"
else:
  JobDataAccessModule = ROOTModulePath + "/" + "QJobDataAccess_C.so"

#cmd = "root -q -b -l '{0}(\"{1}\",\"{2}\",\"{3}\",\"{4}\")'".format(HistBuilderModule,InputFileName,InputDataPath,OutputFileName,OutputPath)
#os.system(cmd)
ROOT.gROOT.ProcessLine(".L {0}".format(JobDataAccessModule))
from ROOT import QJobDataAccess

#EntryId = 1
#CaloId = 10
#CrystalId = 21

while 1:
  Arguments = input("Entering Calo and Crystal Ids : ")
  if not Arguments:
    break
  CaloId, CrystalId = Arguments.split()
  CaloId = int(CaloId)
  CrystalId = int(CrystalId)

  JobHists = QJobDataAccess(CaloId,InputFileName,InputDataPath,OutputFileName,OutputPath)

  c1 = ROOT.TCanvas("c1","c1",0,0,1200,800)
  c1.Divide(9,6)

  for i in range(54):
    c1.cd(i+1)
    JobHists[1][i].Draw()
    JobHists[2][i].Draw("same")
    JobHists[2][i].SetLineColor(ROOT.kRed)
    JobHists[3][i].Draw("same")
    JobHists[3][i].SetLineColor(ROOT.kGreen)
    JobHists[3][i].SetLineWidth(5)

  c1.Update()

  c2 = ROOT.TCanvas("c2","c2",0,0,1200,800)

  JobHists[0][0].Draw()
  JobHists[0][1].Draw("same")
  JobHists[0][1].SetLineColor(ROOT.kRed)
  JobHists[0][2].Draw("same")
  JobHists[0][2].SetLineColor(ROOT.kGreen)
  JobHists[0][2].SetLineWidth(5)

  c2.Update()

  c3 = ROOT.TCanvas("c3","c3",0,0,1200,800)

  JobHists[1][CrystalId].Draw()
  JobHists[2][CrystalId].Draw("same")
  JobHists[2][CrystalId].SetLineColor(ROOT.kRed)
  JobHists[3][CrystalId].Draw("same")
  JobHists[3][CrystalId].SetLineColor(ROOT.kGreen)
  JobHists[3][CrystalId].SetLineWidth(5)

  c3.Update()

  c4 = ROOT.TCanvas("c4","c4",0,0,1200,800)

  JobHists[1][CrystalId].Draw()
  JobHists[2][CrystalId].Draw("same")
  JobHists[2][CrystalId].SetLineColor(ROOT.kRed)
  c4.Update()

  c5 = ROOT.TCanvas("c5","c5",0,0,1200,800)

  JobHists[0][3].Draw()
  JobHists[0][3].Draw("same")
  c5.Update()
