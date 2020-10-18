import csv
import time
import sys
from tqdm import tnrange, notebook
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors
import math
import ROOT as rt
from matplotlib import ticker

qF1 = rt.TFile("TruthRootOut_0000.root")
qh1 = qF1.Get("TruthQHist")
nbins=qh1.GetNbinsX()
q1data=np.zeros(nbins)
q2data=np.zeros(nbins)
for idx in range(0,nbins):
    q1data[idx]=qh1.GetBinContent(idx+1)
qF1.Close()
qF2 = rt.TFile("RPRootOut_0000.root")
qh2 = qF2.Get("AnaQHist")
for idx in range(0,nbins):
    q2data[idx]=qh2.GetBinContent(idx+1)
qF2.Close()
fig, ax = plt.subplots()
ax.plot(np.arange(0,nbins)[100:],(q1data-q2data)[100:])
ax.set_title("Truth-RP")
plt.savefig("diff.png", dpi=150)
c1=rt.TCanvas("c1","c1",1920,1080)
print(nbins)
qdraw=rt.TH1F("diff","Truth-RP diff",nbins-100,100,nbins)
for idx in range(0,3000-100):
    # print(idx)
    qdraw.SetBinContent(idx,((q1data-q2data)[100:3000])[idx-1])
c1.Draw()
qdraw.Draw()
c1.Update()
c1.SaveAs("diff_root.png")
for idx in range(0,3000-100):
    # print(idx)
    qdraw.SetBinContent(idx,(q1data[100:3000])[idx-1])
qdraw.SetTitle("Truth sum")
qdraw.Draw()
c1.Update()
c1.SaveAs("truth_sum.png")
for idx in range(0,3000-100):
    # print(idx)
    qdraw.SetBinContent(idx,(q2data[100:3000])[idx-1])
qdraw.SetTitle("RP Q sum")
qdraw.Draw()
c1.Update()
c1.SaveAs("rp_sum.png")
