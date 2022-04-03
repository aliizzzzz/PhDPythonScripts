#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:44:37 2021

@author: alizz
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# %% Inputs

savePlots = False
quality = 450
home = '/Users/alizz/OneDrive - University of Southampton/Shared Folder - Venous Simulation/Transient MI Study/MI Study (IRIDIS Upload)/'
dataLoc = 'reportFiles/'
plotLoc = 'plots/'
keywords = ['surface-coverage','blood-displacement-front','blood-displacement-back','max-velocity']
header = {}
ylabel = {}
ylim = {}

bdLimit = [-4.98153508284348,-4.981535082843515]
for word in keywords:
    if "back" in word:
        header[word] = "Outlet 2 (Back)"
        ylabel[word] = "Blood Outflow (mL/min)"
        ylim[word] = [-10,0]
    elif "front" in word:
        header[word] = "Outlet 1 (Front)"
        ylabel[word] = "Blood Outflow (mL/min)"
        ylim[word] = [-10,0]
    elif 'velocity' in word:
        header[word] = "Maximum Velocity Magnitude"
        ylabel[word] = "Velocity Magnitude (m/s)"
        ylim[word] = [0,3]
    else:
        header[word] = word.replace("-"," ").title()
        ylabel[word] = "% Wall surface area covered by foam"
        ylim[word] = [0,60]
rhoBlood = 1000 #kg/m3
data = {}
for word in keywords:
    data[word] = {}

# %% Read Report Files

directory = os.fsencode(home+dataLoc)

# fileContour = open(cDir.full+'foamContour-courant0p85-local0p01-10mlmin-0250')
# for line in fileContour:
directory = os.fsencode(home+dataLoc)
directory_files = sorted(os.listdir(directory))
meshX = []
finalSC = []
for file in directory_files:
    filename = os.fsdecode(file)
    if not filename.endswith(".out"):
        continue
    key1 = filename[8:-4]
    key2 = filename[2:5]
    if key2 not in data[key1].keys():
        data[key1][key2]     = {}
    if key2 not in data[key1].keys():
        data[key1][key2]  = {}
    if key2 not in data[key1].keys():
        data[key1][key2]   = {}
    for word in keywords:
        if word in filename:

            with open(home+dataLoc+filename) as file:
                lineNo = 0
                t =[]
                y =[]
                for line in file.readlines():
                    lineNo += 1
                    if lineNo < 4:
                        continue
                    t.append(float(line.split(" ")[2][:-1]))
                    y.append(float(line.split(" ")[1]))
                if "blood" in key1:
                    y = [x*6e7/rhoBlood for x in y] #kg/s -> mL/min
                if "surface" in key1:
                    meshX.append(float(key2))
                    y = [100*x for x in y]
                    finalSC.append(y[-1])
                data[key1][key2]['time']    = t
                data[key1][key2][key1]      = y

# %% Plot Graphs
markers = ['v','x','v','<','>','s','o','^']
for selection in keywords:
    xlabel = "Flow Time (s)"
    title = "Mesh Independence of " + header[selection]
    fig = plt.figure(figsize=[12, 9])
    fig.suptitle(title,weight='bold', fontsize = 18)
    plt.style.use('ggplot')
    i = 0
    for key in data[selection].keys():
        plt.plot(data[selection][key]['time'],data[selection][key][selection],label = key + " $\mathregular{\mu}$m",linestyle='none',marker = markers[i], markersize=6)
        plt.legend(prop={'family': 'monospace', 'size': 14},
                     handletextpad=0.3)
        plt.xlabel(xlabel,fontsize=18, labelpad=15)
        plt.ylabel(ylabel[selection],fontsize=18, labelpad=15)
        plt.grid(True, which='both')
        plt.grid(linestyle=':', linewidth=0.5,which='minor')
        plt.ylim(ylim[selection])
        plt.xlim([0,round(max(data[selection][key]['time']))])
        plt.xticks(fontsize=12)
        ax = plt.gca()
        ax.tick_params(which='both', pad=12, labelsize=14)
        i += 1
    if 'surface' in selection:
        fig2 = plt.figure(figsize=[12, 9])
        title2 = "Mesh Independence of " + header[selection] + " (Zoomed)"
        fig2.suptitle(title2,weight='bold', fontsize = 18)
        plt.style.use('ggplot')
        i = 0
        for key in data[selection].keys():
            plt.plot(data[selection][key]['time'],data[selection][key][selection],label = key + " $\mathregular{\mu}$m",linestyle='none',marker = markers[i], markersize=6)
            plt.legend(prop={'family': 'monospace', 'size': 14},
                         handletextpad=0.3)
            plt.xlabel(xlabel,fontsize=18, labelpad=15)
            plt.ylabel(ylabel[selection],fontsize=18, labelpad=15)
            plt.grid(True, which='both')
            plt.grid(linestyle=':', linewidth=0.5,which='minor')
            plt.ylim([15,55])
            plt.xlim([3,round(max(data[selection][key]['time']))])
            plt.xticks(fontsize=12)
            ax = plt.gca()
            ax.tick_params(which='both', pad=12, labelsize=14)
            i += 1
    if 'blood' in selection:
        ylim[selection] = [ylim[selection][0]*rhoBlood/6e4,ylim[selection][1]*rhoBlood/6e4]
        yax2 = plt.twinx()
        yax2.grid(False, which='both')
        yax2.set_ylabel("g/s", fontsize=18, labelpad=15)
        yax2.tick_params(axis='both', labelsize=14)
        yax2.set_ylim(ylim[selection])
        yax2.set_yticks([round(x,2) for x in np.linspace(min(ylim[selection]),max(ylim[selection]),6)])
        # yax2.ticklabel_format(axis='y', style='sci', useMathText=True, scilimits = (0,0))

    if savePlots == True:
        fig.savefig((home + plotLoc + title + '.png'),dpi=quality, transparent = False)
        fig2.savefig((home + plotLoc + title2 + '.png'),dpi=quality, transparent = False)

xlabel = "Element Size ($\mathregular{\mu}$m)"
title = "Maximum Wall Surface Coverage"

# %% Maximum Wall Surface Coverage

fig = plt.figure(figsize=[12, 9])
fig.suptitle(title,weight='bold', fontsize = 18)
slope, intercept, r_value, p_value, std_err = linregress(meshX,finalSC)
annotation = '$\mathit{y={%.4f}x+{%.4f}}$'%(slope,intercept)+'\n'+'$\mathit{{R^2}=%0.4f}$'%(r_value)
plt.plot(meshX,finalSC,marker = '+', color='r', markersize = 6,linestyle='none')
plt.plot(np.linspace(0,500,501),(slope*np.linspace(0,500,501))+intercept, linestyle ='-.',color='g', linewidth=1)
plt.xlabel(xlabel,fontsize=18, labelpad=15)
plt.ylabel('Max. % Surface Coverage',fontsize=18, labelpad=15)
plt.grid(True, which='both')
plt.grid(linestyle=':', linewidth=0.5,which='minor')
plt.ylim([0,60])
plt.xlim([0,500])
plt.xticks(fontsize=12)
plt.annotate(annotation,(300,40),color='g',fontsize=16 )
ax = plt.gca()
ax.tick_params(which='both', pad=12, labelsize=14)

if savePlots == True:
    fig.savefig((home + plotLoc + title + '.png'),dpi=quality, transparent = False)