#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 16:38:28 2019

@author: Alireza Meghdadi
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import sem

# %% Inputs

# Output Parameters:
saveDpPlots     = False
saveResults     = False
quality         = 300
# Offset from the point of minimum gradient to be taken as second plateau point:
PrimaryOffset   = -10
RepeatNum       = 3

# Directory:
class cDir:
    """Holds directory information about each experiment which then will be written to file names."""
    dataDir     = '/Users/alizz/Google Drive (alireza.meghdadi@gmail.com)/Southampton - GoogleDrive/'
    dataDir     += 'Experiments/Pipe Viscometry/Data/'
    pipeDir     = '2.48 mm PTFE/'
    tecDir      = 'TSS/'
    lgDir       = '1-5/'
    spacer      = ' | '
    full        = dataDir + pipeDir + tecDir + lgDir

# Series Information:
class info:
    """Hold information about the experiments."""
    pipe        = cDir.pipeDir[:-1]
    technique   = cDir.tecDir[:-1]
    lgRatio     = cDir.lgDir[:-1]

# Pipe Information:
diameter        = float(cDir.pipeDir[:4])   # mm
lengths         = {'2.48': 34,
                   '4.48': 15}
connector       = {'2.48': 0,
                   '4.48': 2*(0.263+0.066658313)*1E-6}
length          = lengths[cDir.pipeDir[:4]]                        # cm
TubeVol         = (np.pi*((diameter/2000)**2)*length/100)+connector[cDir.pipeDir[:4]]

# %% Functions
def findTruePtressures(recordedSignal):
    """
    Returns the calibrated (true) pressure values
    """
    correctedSignal = 1.0325*(recordedSignal) + 17.573
    return correctedSignal

def findPlateau(i,o,t,indices,off):
    """
    Finds two points that flank the pressure plateau.
        i       : Inlet pressure array
        o       : Outlet pressure array
        t       : Time array
        indices : Index of the first graphical input
        off     : offset

        Returns the index of the starting and ending points of plateau as a tuple
    """
    dp               = i - o
    dp1              = dp[indices[0]:indices[1]]
    gradInlet        = np.gradient(i[indices[2]:indices[3]], t[indices[2]:indices[3]])
    gradOutlet       = np.gradient(o[indices[2]:indices[3]], t[indices[2]:indices[3]])

    idx1             = np.argwhere(dp1 == max(dp1))[0]
    idx2             = min(np.argwhere(gradInlet == min(gradInlet))[0],
                           np.argwhere(gradOutlet == min(gradOutlet))[0])

    return (indices[0]+idx1[0],indices[2]+idx2[0]+off)


def calcRepeatAverage(values, givstd=[]):
    threes      = []
    averaged    = []
    Sems        = []
    Stds        = []
    k           = 0
    t           = 0
    for val in values:
        if k < RepeatNum:
            threes.append(val)
            k += 1
            t += 1
            if k == RepeatNum:
                averaged.append(np.mean(threes))
                Sems.append(sem(threes))
                Stds.append(np.sqrt(np.sum(np.square(givstd[t-3:t])))/RepeatNum)
            else:
                continue
        else:
            threes = []
            threes.append(val)
            k =1
            t += 1
    return (averaged, Sems, Stds)


def plotCurves (graph,t,i,o,heading):
    """
    Plots graph of inlet (i) and outlet (o) versus time (t). Takes a figure
    plot and a heading as arguements

    Inputs: graph, t,i, o, heading
    """
    dp = i - o
    graph.plot(t,i,linewidth=1, color='b', label='$\mathregular{P_{in}}$')
    graph.plot(t,o,linewidth=1, color='r', label='$\mathregular{P_{out}}$')
    graph.plot(t,dp,linewidth=1, color='g', label='$\mathregular{\Delta}$P')
    graph.legend(fontsize = 12)
    graph.xaxis.set_major_locator(plt.MultipleLocator((np.round((max(t)+100)/10,-1))))
    graph.yaxis.set_major_locator(plt.MaxNLocator(10))
    graph.minorticks_on()
    graph.grid(True, which='both')
    graph.grid(linestyle=':',linewidth=0.5,which='minor')
    graph.xaxis.set_minor_locator(plt.MultipleLocator(5))
    graph.set_ylim(ylimit)
    graph.set_xlim(xlimit)
    graph.tick_params(axis='both', labelsize=14)
    graph.set_xlabel('Time (s)', fontsize=16, labelpad=5)
    graph.set_ylabel('Pressure (mBar)', fontsize=16, labelpad=5)
    graph.title.set_text(heading)
    graph.title.set_fontsize(22)
    graph.title.set_position(titlepos)


def calcVisc(dp,error,f,L,d):
    """
    Inputs: dp, error,f, L, d

    Returns:
        [0]:    tw
        [1]:    twError
        [2]:    viscosity
        [3]:    viscoisityError
    """
    absolerr                = 2*error  # 2 standard deviations as error
    relerror                = absolerr/dp

    tw                      = np.empty(len(dp))
    twError                 = np.empty(len(error))
    viscPoiseuille          = np.empty(len(dp))
    viscPoiseuilleError     = np.empty(len(error))
    for i in range(len(dp)):
        np.put(viscPoiseuille,i,(np.pi*dp[i]*d**4)/(128*f[i]*L))      # Viscosity in Pa.s
        np.put(viscPoiseuilleError,i,relerror[i]*viscPoiseuille[i])
        np.put(tw,i,dp[i]*d/(4*L))                                    # Wall shear stress in Pa
        np.put(twError,i,relerror[i]*tw[i])                           # Wall shear stress error
    return (tw,twError,viscPoiseuille,viscPoiseuilleError)

# %% Read CSV
csvs        = []
Q           = []
dpAve       = []
dpStd       = []
dpMin       = []
dpMax       = []
zeroInlet   = 0
zeroOutlet  = 0

directory = os.fsencode(cDir.full)
directory_files = sorted(os.listdir(directory))

for file in directory_files:
    filename = os.fsdecode(file)
    # Calculate zero baseline:
    if filename.endswith(".csv") and filename.startswith("zero"):
        with open(cDir.full + filename) as csvFile:
            reader = csv.reader(csvFile)
            time, inlet, outlet = [],[],[]

            for row in reader:
                time.append(row[0])
                inlet.append(row[2])
                outlet.append(row[4])
        zeroInlet   = np.mean(np.asarray(list(map(float,inlet[1:]))))
        zeroOutlet  = np.mean(np.asarray(list(map(float,outlet[1:]))))
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
        # Calculate offsets
        if len(csvs) <= 6:
            offset = PrimaryOffset
        elif 6 < len(csvs) <= 12:
            offset = PrimaryOffset + 2
        elif 12 < len(csvs) <= 18:
            offset = PrimaryOffset + 3
        elif 18 < len(csvs) <= 24:
            offset = PrimaryOffset + 4
        else:
            offset = PrimaryOffset + 5
        with open(cDir.full + filename) as csvFile:
            reader = csv.reader(csvFile)
            time, inlet, outlet = [],[],[]

            for row in reader:
                time.append(row[0])
                inlet.append(row[2])
                outlet.append(row[4])

        time    = np.asarray(list(map(float,time[1:])))
        inlet   = np.asarray(list(map(float,inlet[1:])))-zeroInlet
        outlet  = np.asarray(list(map(float,outlet[1:])))-zeroOutlet

        # repeat  = int(filename[-5:-4])
        # flow    = float(filename[0:2])
        AcqFreq = round(len(inlet)/max(time))

    # %% Smoothing
        sIn     = inlet
        sOut    = outlet

        # Forward and backward digital filter
        # b, a    = signal.ellip(3, 0.01, 120, 0.03125)
        # sIn     = signal.filtfilt(b, a, sIn, method="gust")
        # sOut    = signal.filtfilt(b, a, sOut, method="gust")

        # Savitzky-Golay filter:
        a       = 15
        sIn     = signal.savgol_filter(inlet,a,2)
        sOut    = signal.savgol_filter(outlet,a,2)

    # %% Plot and Trim ROI
        maxTimeRange     = np.round(max(time)+5,-1)
        maxInletRange    = np.round(max(sIn)+20,-1)
        xlimit      = [0,maxTimeRange]
        ylimit      = [0,maxInletRange]
        title       = filename[0:3] + '$\mathregular{mL.min^{-1}}$ #' + filename[-5:-4]
        title      += ' - Ac. Freq. %.2f Hz, Offset = %.0f' % (AcqFreq, offset)
        titlepos        = [0.5,1.1]

        fig = plt.figure(figsize=[15, 13])
        fig.subplots_adjust(wspace=0.3, hspace=0.55)
        fig.suptitle(title,weight='bold', fontsize = 18)
        plt.style.use('ggplot')

        g1 = fig.add_subplot(221)
        g2 = fig.add_subplot(222)
        g3 = fig.add_subplot(223)
        g4 = fig.add_subplot(224)

        plotCurves(g1,time,inlet,outlet,'Original')
        plotCurves(g2,time,sIn,sOut, 'Smoothed')

        i               = len(csvs) - 1             # position of data for current file
        Q.append(float(filename[0:2])/(6*10**7))                    # Flowrates in m3/s
        tubePrimeIndx   =(np.abs(time - (TubeVol/Q[i]))).argmin()   # Idx of time when tube is primed
        vertical = np.linspace(0,ylimit[1],num=10,dtype=float)
        
        # 2s added to ensure tube is primed:
        g2.axvspan(0, time[tubePrimeIndx]+2, alpha=0.15, color='k', label = 'Tube Primed')
        g2.legend(fontsize=12)

        pts     = plt.ginput(4)

        indices = [
            (np.abs(time - pts[0][0])).argmin(),
            (np.abs(time - pts[1][0])).argmin(),
            (np.abs(time - pts[2][0])).argmin(),
            (np.abs(time - pts[3][0])).argmin()
            ]

        plateauIdx = findPlateau(sIn,sOut,time,indices,offset)

        g2.plot(time[plateauIdx[0]],sIn[plateauIdx[0]],marker='+',markersize=12,color='g',mew=2)
        g2.plot(time[plateauIdx[0]],sOut[plateauIdx[0]],marker='+',markersize=12,color='g',mew=2)
        g2.plot(time[plateauIdx[0]],sIn[plateauIdx[0]]-sOut[plateauIdx[0]],marker='+',markersize=12,
                color='g',mew=2)
        g2.axvspan(time[indices[0]], time[indices[1]], alpha=0.20, color='g',
                   label = 'Local P1 Region')
        g2.legend(fontsize=12, ncol=2)

        plotCurves(g3,time[plateauIdx[0]:],sIn[plateauIdx[0]:],sOut[plateauIdx[0]:], 'LHS Trimmed')
        g3.plot(time[plateauIdx[1]],sIn[plateauIdx[1]],marker='+',markersize=12,color='g',mew=2)
        g3.plot(time[plateauIdx[1]],sOut[plateauIdx[1]],marker='+',markersize=12,color='g',mew=2)
        g3.plot(time[plateauIdx[1]],sIn[plateauIdx[1]]-sOut[plateauIdx[1]],marker='+',markersize=12,
                color='g',mew=2)
        g3.axvspan(time[indices[2]], time[indices[3]], alpha=0.20, color='g',
                   label = 'Local P2 Region')
        g3.legend(fontsize=12)

# %% Save Parameters

        # Pressure difference in Pa
        dp              = np.subtract(sIn[plateauIdx[0]:plateauIdx[1]],
                                      sOut[plateauIdx[0]:plateauIdx[1]])*100

        plotCurves(g4,time[plateauIdx[0]:plateauIdx[1]],sIn[plateauIdx[0]:plateauIdx[1]],
                   sOut[plateauIdx[0]:plateauIdx[1]], 'LHS and RHS Trimmed')

        dpAve.append(np.average(dp))
        dpStd.append(np.std(dp))
        dpMax.append(max(dp))
        dpMin.append(min(dp))

    # %% Save Figures
        folder = 'Figures/' + info.pipe + cDir.spacer + info.technique + cDir.spacer + info.lgRatio
        if saveDpPlots == True:
            if os.path.exists(cDir.dataDir+folder) == False:
                os.mkdir(cDir.dataDir+folder)
            fig.savefig((cDir.dataDir+folder + '/' + filename[0:-4] +'.png'),dpi=quality,
                        transparent = False)
    else:
        continue

# %% Final Calculations
d                       = diameter/1000                 # Diameter in m
L                       = length/100                    # Length in m
# Shear Rate
flowrates               = np.empty(int(len(Q)/3))
obsShear                = np.empty(int(len(Q)/3))
for i in range(len(Q)):
    if (i % 3 == 0):
        np.put(flowrates,i/3,Q[i])                      # Flowrates in m3/s
Q = np.empty(len(flowrates))
for i in range(len(Q)):
    np.put(obsShear,i,(32*flowrates[i])/(np.pi*d**3))   # Observed Shear rate in 1/s
    np.put(Q,i,(60000000*flowrates[i]))                 # Flowrates in mL/min

# Average Viscosity
dpAveRep                = np.array(calcRepeatAverage(dpAve)[0])
dpAveStd                = np.array(calcRepeatAverage(dpAve,dpStd)[2])
tw                      = calcVisc(dpAveRep,dpAveStd,flowrates,L,d)[0]
twStd                   = calcVisc(dpAveRep,dpAveStd,flowrates,L,d)[1]
viscPoiseuille          = calcVisc(dpAveRep,dpAveStd,flowrates,L,d)[2]
viscPoiseuilleStd       = calcVisc(dpAveRep,dpAveStd,flowrates,L,d)[3]

# Minimum Viscosity
dpMinRep                = np.array(calcRepeatAverage(dpMin)[0])
dpMinSem                = np.array(calcRepeatAverage(dpMin)[2])
twMin                   = calcVisc(dpMinRep,dpMinSem,flowrates,L,d)[0]
twSemMin                = calcVisc(dpMinRep,dpMinSem,flowrates,L,d)[1]
viscPoiseuilleMin       = calcVisc(dpMinRep,dpMinSem,flowrates,L,d)[2]
viscPoiseuilleSemMin    = calcVisc(dpMinRep,dpMinSem,flowrates,L,d)[3]

# Maximum Viscosity
dpMaxRep                = np.array(calcRepeatAverage(dpMax)[0])
dpMaxSem                = np.array(calcRepeatAverage(dpMax)[2])
twMax                   = calcVisc(dpMaxRep,dpMaxSem,flowrates,L,d)[0]
twSemMax                = calcVisc(dpMaxRep,dpMaxSem,flowrates,L,d)[1]
viscPoiseuilleMax       = calcVisc(dpMaxRep,dpMaxSem,flowrates,L,d)[2]
viscPoiseuilleSemMax    = calcVisc(dpMaxRep,dpMaxSem,flowrates,L,d)[3]

# %% Write Results to CSV
if saveResults == True:
    resultsFile =info.pipe + cDir.spacer + info.technique + cDir.spacer + info.lgRatio +'.csv'
    if os.path.exists(cDir.dataDir + 'Preprocessed Results') == False:
        os.mkdir(cDir.dataDir + 'Preprocessed Results')
    with open((cDir.dataDir + 'Preprocessed Results/' + resultsFile),'w',newline='') as csvfile:
        fields = ['flowrate (ml/min)'               ,
                  'flowrate (m3/s)'                 ,
                  'dPAve (mBar)'                    ,
                  'dPStd'                           ,
                  'poiseuilleViscosity (Pa.s)'      ,
                  'poiseuilleViscosityMin (Pa.s)'   ,
                  'poiseuilleViscosityMax (Pa.s)'   ,
                  'poisViscStd'                     ,
                  'poisViscSemMin'                  ,
                  'poisViscSemMax'                  ,
                  'observedShearRate (1/s)'         ,
                  'wallShearStress (Pa)'            ,
                  'wallShearStressMin (Pa)'         ,
                  'wallShearStressMax (Pa)'         ,
                  'shearStressStd'                  ,
                  'shearStressSemMin'               ,
                  'shearStressSemMax'               ]

        writer = csv.DictWriter(csvfile, fieldnames = fields)
        writer.writeheader()
        for i in range(len(dpAveRep)):
            writer.writerow({
                    'flowrate (ml/min)'             :   Q[i]                    ,
                    'flowrate (m3/s)'               :   flowrates[i]            ,
                    'dPAve (mBar)'                  :   dpAveRep[i]/100         ,
                    'dPStd'                         :   dpAveStd[i]/100         ,
                    'poiseuilleViscosity (Pa.s)'    :   viscPoiseuille[i]       ,
                    'poiseuilleViscosityMin (Pa.s)' :   viscPoiseuilleMin[i]    ,
                    'poiseuilleViscosityMax (Pa.s)' :   viscPoiseuilleMax[i]    ,
                    'poisViscStd'                   :   viscPoiseuilleStd[i]    ,
                    'poisViscSemMin'                :   viscPoiseuilleSemMin[i] ,
                    'poisViscSemMax'                :   viscPoiseuilleMax[i]    ,
                    'observedShearRate (1/s)'       :   obsShear[i]             ,
                    'wallShearStress (Pa)'          :   tw[i]                   ,
                    'wallShearStressMin (Pa)'       :   twMin[i]                ,
                    'wallShearStressMax (Pa)'       :   twMax[i]                ,
                    'shearStressStd'                :   twStd[i]                ,
                    'shearStressSemMin'             :   twSemMin[i]             ,
                    'shearStressSemMax'             :   twMax[i]                })
