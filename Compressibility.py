#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:27:08 2021

@author: alizz
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress

plotp1p2 = False
plotCurves = False
plotBulkMod = False
plotReg = True
saveBulkMod = False
saveReg = True
saveResults = True
quality = 450

home = "/Users/alizz/OneDrive - University of Southampton/Shared Folder - Venous Simulation/Experiments/Foam Compressibility/"
foams = ['DSS-1-3', 'DSS-1-4', 'DSS-1-5', 'TSS-1-4', 'TSS-1-5']
savePath = home + "Figures/"
csvs        = []
rho1, rho2, time, pressure, pressure_onefilt = {}, {}, {}, {}, {}
rhoWater    = 997 # kg/m3
rhoAir      = 1.225 # kg/m3
for foam in foams:
    time[foam]      = {}
    pressure[foam]  = {}
    pressure_onefilt[foam] = {}
    rho1[foam]= (1/(1+float(foam[-1]))*rhoWater)+((1-(1/(1+float(foam[-1]))))*rhoAir)

# %% Read files and store the data
for foam in foams:
    path = home + foam + "/"
    directory = os.fsencode(path)
    directory_files = sorted(os.listdir(directory))

    for file in directory_files:
        filename = os.fsdecode(file)
        # Calculate zero baseline:
        if filename.endswith(".csv") and filename.startswith("zero"):
            with open(path + filename) as csvFile:
                reader = csv.reader(csvFile)
                time_zero, pressure_zero = [],[]

                for row in reader:
                    time_zero.append(row[0])
                    pressure_zero.append(row[2])
            zero   = np.mean(np.asarray(list(map(float,pressure_zero[1:]))))
            break
        else:
            continue
    for file in directory_files:
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename.startswith("zero"):
            continue
        # Open CSV files:
        if filename.endswith(".csv"):
            csvs.append(filename)
            with open(path + filename) as csvFile:
                reader = csv.reader(csvFile)
                t, p = [],[]

                for row in reader:
                    t.append(row[0])
                    p.append(row[2])
            time[foam][filename[-5]]    = np.asarray(list(map(float,t[1:])))
            pressure[foam][filename[-5]]   = signal.savgol_filter(signal.savgol_filter(np.asarray(list(map(float,p[1:])))-zero,15,3),51,2)
            pressure_onefilt[foam][filename[-5]]   = signal.savgol_filter(np.asarray(list(map(float,p[1:])))-zero,15,3)
            # b, a    = signal.ellip(3, 0.01, 60, 0.03125)
            # pressure[foam][filename[-5]] = signal.filtfilt(b, a, np.asarray(list(map(float,pressure[foam][filename[-5]][1:])))-zero,method="gust")
# %% Obtain indices of plateau region
p1indices = {}
p2indices = {}
for foam in foams:
    p1indices[foam]   = {}
    p2indices[foam]   = {}

for i in range(1,4):
    rep = str(i)
    for foam in foams:
        for value in np.gradient(pressure[foam][rep]):
            if np.argwhere(np.gradient(pressure[foam][rep]) == value)[0][0] == 0:
                continue
            if value > 1.5:
                p1indices[foam][rep] = np.argwhere(np.gradient(pressure[foam][rep]) == value)[0][0]
                break
    for foam in foams:
        for value in np.gradient(pressure[foam][rep])[::-1]:
            if np.argwhere(np.gradient(pressure[foam][rep])[::-1] == value)[0][0] == 0:
                continue
            if value > 2:
                p2indices[foam][rep] = np.argwhere(np.gradient(pressure[foam][rep]) == value)[0][0]+1
                break

# %% Calculate average pressure values
p1Ave       = {}
p2Ave       = {}
p1StdAve    = {}
p2StdAve    = {}

for foam in foams:
    p1std = []
    p2std = []
    p1  = []
    p2  = []
    for i in range(1,4):
        rep = str(i)
        p1.append(np.average(pressure[foam][rep][:p1indices[foam][rep]]))
        p2.append(np.average(pressure[foam][rep][p2indices[foam][rep]:]))
        p1std.append(np.std(pressure[foam][rep][:p1indices[foam][rep]]))
        p2std.append(np.std(pressure[foam][rep][p2indices[foam][rep]:]))
    p1Ave[foam] = np.average(p1)
    p2Ave[foam] = np.average(p2)
    p1StdAve[foam] = (np.sqrt(np.sum(np.square(p1std))))/3
    p2StdAve[foam] = (np.sqrt(np.sum(np.square(p2std))))/3

# %% Plot P1 and P2
plt.style.use('ggplot')
if plotp1p2 == True:
    for foam in foams:
        fig = plt.figure(figsize=[8, 6])
        fig.suptitle(foam,weight='bold', fontsize = 18)
        for i in range(1,4):
            rep = str(i)
            g1 = plt.plot(time[foam][rep][:p1indices[foam][rep]], pressure[foam][rep][:p1indices[foam][rep]])
            plt.plot(time[foam][rep][p2indices[foam][rep]:], pressure[foam][rep][p2indices[foam][rep]:], color = g1[-1].get_color())
            plt.plot(time[foam][rep][p1indices[foam][rep]:p2indices[foam][rep]],pressure[foam][rep][p1indices[foam][rep]:p2indices[foam][rep]], label = rep)
            plt.xlabel('Time (s)')
            plt.ylabel('Pressure (mBar)')
            plt.legend()
#%% P - V graph
vt = 10                                     # mL
vSiliconTubes = 2*(2*(np.pi*(0.206/2)**2))  # mL
vMainTube = 15*(np.pi*(0.448/2)**2)         # mL
vSensor = 0.07                              # mL
vSystem = vMainTube+vSiliconTubes+vSensor   # mL, volume of the system without the syringe (PTFE+2*C-Flex+sensor)
v1 = 10 - (3-vSystem)                       # mL
v2 = v1 - 3                                 # mL
q = 500/60                                  # mL/min

if plotCurves == True:
    for foam in foams:
        fig = plt.figure(figsize=[8, 6])
        fig.suptitle(foam,weight='bold', fontsize = 18)
        for i in range(1,4):
            rep = str(i)
            plt.plot(np.linspace(v1,v2,len(pressure[foam][rep][p1indices[foam][rep]:p2indices[foam][rep]])),pressure[foam][rep][p1indices[foam][rep]:p2indices[foam][rep]])
            plt.xlabel('Volume (mL)')
            plt.ylabel('Pressure (mBar)')
for foam in foams:
    rho2[foam] = rho1[foam]*v1/v2
# %% Bulk Modulus Calculations
bulkModuli = {}
bulkModuli_onefilt = {}
b1indices = {}
b2indices = {}
for foam in foams:
    b1indices[foam]   = {}
    b2indices[foam]   = {}

bulkModuliOverall = {}
for foam in foams:
    bulkModuliOverall[foam] = -v1*((p2Ave[foam]-p1Ave[foam])/(v2-v1))*100 # Pascals

for foam in foams:
    bulkModuli[foam] = {}
    bulkModuli_onefilt[foam] = {}
    for i in range(1,4):
        rep = str(i)
        bulkModuli[foam][rep]         = signal.savgol_filter(-v1*(np.gradient(pressure[foam][rep][p1indices[foam][rep]:p2indices[foam][rep]])/(np.linspace(v1,v2,len(pressure[foam][rep][p1indices[foam][rep]:p2indices[foam][rep]]))[1]-v1))*100,15,2) # Pascals
        bulkModuli_onefilt[foam][rep] = signal.savgol_filter(-v1*(np.gradient(pressure_onefilt[foam][rep][p1indices[foam][rep]:p2indices[foam][rep]])/(np.linspace(v1,v2,len(pressure_onefilt[foam][rep][p1indices[foam][rep]:p2indices[foam][rep]]))[1]-v1))*100,15,2) # Pascals
# %% Trim bulk modulus data
for i in range(1,4):
    rep = str(i)
    for foam in foams:
        for value in np.gradient(np.gradient(bulkModuli[foam][rep])):
            if np.argwhere(np.gradient(np.gradient(bulkModuli[foam][rep])) == value)[0][0] == 0:
                continue
            if value > 100:
                b1indices[foam][rep] = np.argwhere(np.gradient(np.gradient(bulkModuli[foam][rep])) == value)[0][0] + 15
                break
    for foam in foams:
        for value in np.gradient(np.gradient(bulkModuli[foam][rep]))[::-1]:
            if np.argwhere(np.gradient(np.gradient(bulkModuli[foam][rep]))[::-1] == value)[0][0] == 0:
                continue
            if value > 100:
                b2indices[foam][rep] = np.argwhere(np.gradient(np.gradient(bulkModuli[foam][rep])) == value)[0][0] - 10
                break
b1, b2, pi1, pi2 = {}, {}, {}, {}
bulkModuliAve, pressureAve = {}, {}
for foam in foams:
        b1[foam] = min(b1indices[foam]['1'],b1indices[foam]['2'],b1indices[foam]['3'])
        b2[foam] = min(b2indices[foam]['1'],b2indices[foam]['2'],b2indices[foam]['3'])
        pi1[foam] = min(p1indices[foam]['1'],p1indices[foam]['2'],p1indices[foam]['3'])
        pi2[foam] = min(p2indices[foam]['1'],p2indices[foam]['2'],p2indices[foam]['3'])
        bulkModuliAve[foam] = np.mean([bulkModuli[foam]['1'][b1[foam]:b2[foam]],bulkModuli[foam]['2'][b1[foam]:b2[foam]],bulkModuli[foam]['3'][b1[foam]:b2[foam]]],axis=0)
        pressureAve[foam] = np.mean([pressure[foam]['1'][pi1[foam]:pi2[foam]][b1[foam]:b2[foam]],pressure[foam]['2'][pi1[foam]:pi2[foam]][b1[foam]:b2[foam]],pressure[foam]['3'][pi1[foam]:pi2[foam]][b1[foam]:b2[foam]]],axis=0)

# %% Plot K - P
if saveBulkMod == True:
    plotBulkMod = True

kp_slope, kp_intercept, kp_r, kp_p, kp_slope_err, kp_intercept_err = {}, {}, {}, {}, {}, {}
for foam in foams:
    result = linregress([100*x for x in pressureAve[foam]],bulkModuliAve[foam])
    # result = linregress(dp[foam],bulkModuliAve[foam][1:])
    kp_slope[foam], kp_intercept[foam], kp_r[foam], kp_p[foam], kp_slope_err[foam] = result
    kp_intercept_err[foam] = result.intercept_stderr
   
if plotBulkMod == True:
    for foam in foams:
        fig = plt.figure(figsize=[18, 13])
        fig.suptitle(foam,weight='bold', fontsize = 18)
        # plt.plot(100*np.linspace(pressure[foam][rep][pi1[foam]:pi2[foam]][b1[foam]],pressure[foam][rep][pi1[foam]:pi2[foam]][b2[foam]],10),[bulkModuliOverall[foam] for i in range(0,10)],label = "Overal K", linestyle='None', marker="x")
        plt.plot(100*pressureAve[foam],bulkModuliAve[foam], label = "Average")
        label = '$\mathit{y={%.4f}x{%+.2f}}$'%(kp_slope[foam],kp_intercept[foam])+', $\mathit{{R^2}=%0.4f}$'%((kp_r[foam])**2)
        plt.plot(np.linspace(0,100*pressureAve[foam][-1],10),(kp_slope[foam]*np.linspace(0,100*pressureAve[foam][-1],10))+kp_intercept[foam],linestyle="-.", color = 'g', label = label)
        for i in range(1,4):
            rep = str(i)
            plt.plot(100*pressure[foam][rep][pi1[foam]:pi2[foam]][b1[foam]:b2[foam]],bulkModuli[foam][rep][b1[foam]:b2[foam]], label = "#"+ rep + " 2x SG Filters", alpha=0.75, linestyle=":")
            plt.plot(100*pressure_onefilt[foam][rep][pi1[foam]:pi2[foam]][b1[foam]:b2[foam]],bulkModuli_onefilt[foam][rep][b1[foam]:b2[foam]], label = "#"+ rep + " 1x SG Filter", alpha=0.25, linestyle="-.")
            plt.legend(fontsize=14, loc='lower right', prop={'family': 'monospace', 'size': 18}, handletextpad = 0.8)
            plt.xlabel("Gauge Pressure (Pa)",fontsize=18, labelpad=15)
            plt.ylabel("Bulk Modulus (Pa)",fontsize=18, labelpad=15)
            plt.grid(True, which='both')
            plt.grid(linestyle=':', linewidth=0.5,which='minor')
            plt.ylim([0,275000])
            plt.xlim([0,5000+round(p2Ave[foam]*100,-4)])
            plt.xticks(fontsize=12)
            ax = plt.gca()
            ax.tick_params(which='both', pad=12, labelsize=14)
       
        
            if saveBulkMod == True:
                fig.savefig(savePath+foam+"-K-P.png",dpi = 450, transparent=False)

# %% Linear regression of K - dP
if saveReg == True:
    plotReg = True
dp = {}
# ns = {}
slope, intercept, r_value, p_value, slope_err, intercept_err = {}, {}, {}, {}, {}, {}
for foam in foams:
    dp[foam]=[]
    # ns[foam]=[]
    for i in range(0,len(pressureAve[foam])-1):
        dp[foam].append(100*(pressureAve[foam][i+1]-pressureAve[foam][0])) # Pa
        # ns[foam].append((bulkModuliAve[foam][i+1]-bulkModuliAve[foam][0])/(100*(pressureAve[foam][i+1]-pressureAve[foam][0])))
    # result = linregress([100*x for x in pressureAve[foam]],bulkModuliAve[foam])
    result = linregress(dp[foam],bulkModuliAve[foam][1:])
    slope[foam], intercept[foam], r_value[foam], p_value[foam], slope_err[foam] = result
    intercept_err[foam] = result.intercept_stderr

fig = plt.figure(figsize=[18, 13])
# fig.suptitle(foam,weight='bold', fontsize = 18)
if plotReg == True:
    colors = []
    for foam in foams:
        g = plt.plot(dp[foam],bulkModuliAve[foam][1:],linewidth=0.75, label = '$\mathregular{\Delta}P_{Ave}$ (' + foam[:-4] +' ' + foam[4:].replace('-',':') + ')')
        colors.append(g[-1].get_color())
        # plt.plot([100*x for x in pressureAve[foam]],bulkModuliAve[foam],linewidth=0.75, label = '$\mathregular{P_{Ave}}$')
        # plt.plot(np.linspace(0,dp[foam][-1],10),slope[foam]*np.linspace(0,dp[foam][-1],10)+intercept[foam],linestyle="-.", color = g[-1].get_color(), label = label)
        plt.ylim([100000,280000])
        plt.xlim([0,45000])
        plt.xlabel("Pressure Differential (Pa)",fontsize=18, labelpad=15)
        plt.ylabel("Bulk Modulus (Pa)",fontsize=18, labelpad=15)
        plt.grid(True, which='both')
        plt.grid(linestyle=':', linewidth=0.5,which='minor')
        plt.xticks(fontsize=12)
        ax = plt.gca()
        ax.tick_params(which='both', pad=12, labelsize=14)
    for i, foam in enumerate(foams):
        label = '$\mathit{y={%.4f}x{%+.2f}}$'%(slope[foam],intercept[foam])+', $\mathit{{R^2}=%0.4f}$'%((r_value[foam])**2)
        plt.plot(np.linspace(0,dp[foam][-1],10),slope[foam]*np.linspace(0,dp[foam][-1],10)+intercept[foam],linestyle="-.", color = colors[i], label = label)
plt.legend(fontsize=14, ncol=2, prop={'family': 'monospace', 'size': 18}, handletextpad = 0.8)
if saveReg == True:
    fig.savefig(savePath+"K-dP.png",dpi = 450, transparent=False)

# %% Save Results
if saveResults == True:
    savePath = savePath[:-8] + "Compressibility Parameters/"
    if os.path.exists(savePath) == False:
        os.mkdir(savePath)
    resultsFile = "Compressibility Parameters.csv"
    with open((savePath + resultsFile),'w',newline='') as csvfile:
        fields = ['foam'                                            ,
                  'rho_0'                                           ,
                  'rho'                                             ,
                  'rho_ratio'                                       ,
                  'p_0'                                             ,
                  'p'                                               ,
                  'K_0'                                             ,
                  'K'                                               ,
                  'K-P | slope'                                     ,
                  'K-P | slope_err'                                 ,
                  'K-P | intercept'                                 ,
                  'K-P | intercept_err'                             ,
                  'K-P | r'                                         ,
                  'K-P | p'                                         ,
                  'K-dP | slope'                                    ,
                  'K-dP | slope_err'                                ,
                  'K-dP | intercept'                                ,
                  'K-dP | intercept_err'                            ,
                  'K-dP | r'                                        ,
                  'K-dP | p'                                        ]
        writer = csv.DictWriter(csvfile, fieldnames = fields)
        writer.writeheader()
        for i in range(len(foams)):
            writer.writerow({
                'foam'                                              :       foams[i]                        ,
                'rho_0'                                             :       rho1[foams[i]]                  ,
                'rho'                                               :       rho2[foams[i]]                  ,
                'rho_ratio'                                         :       rho2[foams[i]]/rho1[foams[i]]   ,
                'p_0'                                               :       pressureAve[foams[i]][0]*100    ,
                'p'                                                 :       pressureAve[foams[i]][-1]*100   ,
                'K_0'                                               :       bulkModuliAve[foams[i]][0]      ,
                'K'                                                 :       bulkModuliAve[foams[i]][-1]     ,
                'K-P | slope'                                       :       kp_slope[foams[i]]              ,
                'K-P | slope_err'                                   :       kp_slope_err[foams[i]]          ,
                'K-P | intercept'                                   :       kp_intercept[foams[i]]          ,   
                'K-P | intercept_err'                               :       kp_intercept_err[foams[i]]      ,
                'K-P | r'                                           :       kp_r[foams[i]]                  ,
                'K-P | p'                                           :       kp_p[foams[i]]                  ,
                'K-dP | slope'                                      :       slope[foams[i]]                 ,    
                'K-dP | slope_err'                                  :       slope_err[foams[i]]             ,
                'K-dP | intercept'                                  :       intercept[foams[i]]             ,
                'K-dP | intercept_err'                              :       intercept_err[foams[i]]         ,
                'K-dP | r'                                          :       r_value[foams[i]]               ,
                'K-dP | p'                                          :       p_value[foams[i]]               })
