import os
import ROOT
import numpy as np
import matplotlib.pyplot as plt

ROOTModulePath = os.getenv("QROOTDIR")
InputDataPath = os.getenv("QRAWINPUTDATADIR")
OutputPath = os.getenv("QHISTOGRAMDIR")
InputFileName = "gm2offline_qmethTree.root"
OutputFileName = "QEvent.root"

RecompileModules = True
#RecompileModules = False

#Build the histograms
if RecompileModules:
  EventDataAccessModule = ROOTModulePath + "/" + "QEventDataAccess.C+"
else:
  EventDataAccessModule = ROOTModulePath + "/" + "QEventDataAccess_C.so"

#cmd = "root -q -b -l '{0}(\"{1}\",\"{2}\",\"{3}\",\"{4}\")'".format(HistBuilderModule,InputFileName,InputDataPath,OutputFileName,OutputPath)
#os.system(cmd)
ROOT.gROOT.ProcessLine(".L {0}".format(EventDataAccessModule))
from ROOT import QEventDataAccess

SubRunId = 131

#EntryId = 1
#CaloId = 10
#CrystalId = 21

while 1:
  Arguments = input("Entering Entry, Calo and Crystal Ids : ")
  if not Arguments:
    break
  EntryId, CaloId, CrystalId = Arguments.split()
  EntryId = int(EntryId)
  CaloId = int(CaloId)
  CrystalId = int(CrystalId)

  EventHists = QEventDataAccess(SubRunId,EntryId,CaloId,InputFileName,InputDataPath,OutputFileName,OutputPath)

  c1 = ROOT.TCanvas("c1","c1",0,0,1200,800)
  c1.Divide(9,6)

  for i in range(54):
    c1.cd(i+1)
    EventHists[1][i].Draw()
    EventHists[2][i].Draw("same")
    EventHists[2][i].SetLineColor(ROOT.kRed)
    EventHists[3][i].Draw("same")
    EventHists[3][i].SetLineColor(ROOT.kGreen)
    EventHists[3][i].SetLineWidth(5)
    EventHists[4][i].Scale(100)
#    EventHists[4][i].Draw("HIST Psame")
#    EventHists[4][i].SetMarkerStyle(20)
#    EventHists[4][i].SetMarkerSize(0.7)
#    EventHists[4][i].SetMarkerColor(ROOT.kOrange)

  c1.Update()

  c2 = ROOT.TCanvas("c2","c2",0,0,1200,800)

  EventHists[0][0].Draw()
  EventHists[0][1].Draw("same")
  EventHists[0][1].SetLineColor(ROOT.kRed)
  EventHists[0][2].Draw("same")
  EventHists[0][2].SetLineColor(ROOT.kGreen)
  EventHists[0][2].SetLineWidth(5)
  EventHists[0][3].Scale(100)
  EventHists[0][3].Draw("HIST Psame")
  EventHists[0][3].SetMarkerStyle(20)
  EventHists[0][3].SetMarkerColor(ROOT.kOrange)

  c2.Update()

  c3 = ROOT.TCanvas("c3","c3",0,0,1200,800)

  EventHists[1][CrystalId].Draw()
  EventHists[2][CrystalId].Draw("same")
  EventHists[2][CrystalId].SetLineColor(ROOT.kRed)
  EventHists[3][CrystalId].Draw("same")
  EventHists[3][CrystalId].SetLineColor(ROOT.kGreen)
  EventHists[3][CrystalId].SetLineWidth(5)
  EventHists[4][CrystalId].Draw("HIST Psame")
  EventHists[4][CrystalId].SetMarkerStyle(20)
  EventHists[4][CrystalId].SetMarkerColor(ROOT.kOrange)

  c3.Update()

  c4 = ROOT.TCanvas("c4","c4",0,0,1200,800)

  EventHists[1][CrystalId].Draw()
  EventHists[2][CrystalId].Draw("same")
  EventHists[2][CrystalId].SetLineColor(ROOT.kRed)
  c4.Update()
