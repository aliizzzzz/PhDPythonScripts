#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:35:13 2022

@author: alizz
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress
from scipy.optimize import curve_fit
import cv2
from sys import exit

PLANEAREA = 100*4.48
WALLAREA = 100*np.pi*4.48
home = '/Users/alizz/OneDrive - University of Southampton/Shared Folder - Venous Simulation/'
expPath = 'Experiments/Results/csvs/'
incompressible = 'SimpleTube (PoC)/IRIDIS Directory download (SimpleTube)/Incompressible/'
compressible = 'SimpleTube (PoC)/IRIDIS Directory download (SimpleTube)/Compressible/'
plots = 'SimpleTube (PoC)/IRIDIS Directory download (SimpleTube)/plots/'
foams = ['DSS-1-3/', 'DSS-1-4/', 'DSS-1-5/']
flowrates = ['02mlpm/', '05mlpm/', '10mlpm/']
# flowrates = ['02mlpm/']
reportFiles = ['blood-displacement-back.out', 'blood-displacement-front.out', 'surface-coverage.out']
csvs = {flow: sorted(os.listdir(home+expPath+flow)) for flow in flowrates}


# %% Functions
def getExperimentalData():
    def averager(data, col):
        b, a = signal.ellip(3, 0.01, 120, 0.03125)
        # b, a = signal.butter(8, 0.125)
        # def filt(x): return pd.Series(signal.filtfilt(b, a, x))
        def filt(x): return pd.Series(signal.savgol_filter(x, 10, 1))
        rawdf = pd.concat([filt(data[0][col]),
                           filt(data[1][col]),
                           filt(data[2][col])], axis=1).dropna()
        df = rawdf.apply(pd.DataFrame.describe, axis=1)
        # t = pd.concat([data[0]['Time (s)'], data[1]['Time (s)'], data[2]['Time (s)']],
        # axis=1).dropna()
        return df

    df = {flow: {} for flow in flowrates}
    for flow in flowrates:
        # count = 0
        temp = []
        for csv in csvs[flow]:
            temp.append(pd.read_csv(home+expPath+flow+csv, index_col=0))
            if len(temp) == 3:
                # average foam height, length and area, assign to dataframe
                heightStat = averager(temp, 'FoamHeight (mm)')
                lengthtStat = averager(temp, 'FoamLength (mm)')
                areaStat = averager(temp, 'Area (mm2)')
                time = averager(temp, 'Time (s)')['mean']
                df[flow][csv[:-9]] = pd.concat([time.rename('time'), areaStat['mean'].rename('area'),
                                                areaStat['std'].rename('area-std'),
                                                heightStat['mean'].rename('height'),
                                                heightStat['std'].rename('height-std'),
                                                lengthtStat['mean'].rename('length'),
                                                lengthtStat['std'].rename('length-std')], axis=1)
                temp = []
    return df


def getSimulationData(path):
    df = {}
    df = {flow: {} for flow in flowrates}
    for flow in flowrates:
        for foam in foams:
            # foam = 'DSS-1-3/'
            frontBd = pd.read_csv(home+path+foam+flow+'reportFiles/blood-displacement-front.out',
                                  skiprows=3,
                                  delimiter=' ', names=['time-step', 'bd-front', 'time'])
            backBd = pd.read_csv(home+path+foam+flow+'reportFiles/blood-displacement-back.out',
                                 skiprows=3,
                                 delimiter=' ', names=['time-step', 'bd-back', 'time'])
            frontBd['bd-front'] = frontBd['bd-front'].apply(lambda x: x*6e7/984.59)
            backBd['bd-back'] = backBd['bd-back'].apply(lambda x: x*6e7/984.59)
            surface = pd.read_csv(home+path+foam+flow+'reportFiles/surface-coverage.out', skiprows=3,
                                  delimiter=' ', names=['time-step', 'wall', 'plane', 'time'])
            df[flow][foam[:-1]] = pd.concat([frontBd['time'], surface['plane']*PLANEAREA,
                                             surface['wall']*WALLAREA, frontBd['bd-front'],
                                             backBd['bd-back']], axis=1)
    return df


def getArea(path):
    filename = 'ani.mp4'
    tubeLength = 100  # mm
    threshold = 200
    df = {}
    df = {flow: {} for flow in flowrates}
    for flow in flowrates:
        for foam in foams:
            cap = cv2.VideoCapture(home+path+foam+flow + filename)
            if not cap.isOpened():  # if there are errors exit
                print('Error opening video stream or file')
                exit(1)
            # totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # print(f'\nProcessing {filename}\n{23*"="}')
            # fps = int(cap.get(cv2.CAP_PROP_FPS))
            areas = []
            time = []
            frames = []
            foamHeight = []
            foamLength = []
            while cap.isOpened():
                ret, frame = cap.read()
                cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                if ret is True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if cur_frame == 0:
                        mid = frame.shape[0]//2
                        line = frame[mid, :]
                        calibration_pixels = len(line[line == 228])
                        calibration_mm = float(tubeLength)
                        pixelLength = calibration_mm/calibration_pixels
                        pixelArea = pixelLength**2
                    else:
                        _, thresh = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
                        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,
                                                               cv2.CHAIN_APPROX_SIMPLE)
                        needlePts = (1614, 35) # coordinates of a point that is inside the needle
                        for c in contours[1:]:
                            if cv2.pointPolygonTest(c, needlePts, False) == 1:
                                break
                        left = tuple(c[c[:, :, 0].argmin()][0])
                        right = tuple(c[c[:, :, 0].argmax()][0])
                        bottom = tuple(c[c[:, :, 1].argmax()][0])
                        top = tuple(c[c[:, :, 1].argmin()][0])
                        top = tuple((bottom[0], top[1]))
                        left = tuple((left[0], (right[1]+left[1])//2))
                        right = tuple((right[0], left[1]))
                        contourExtremes = tuple((top, bottom, left, right))
                        foamHeight.append(pixelLength*(contourExtremes[1][1] - contourExtremes[0][1]))
                        foamLength.append(pixelLength*(contourExtremes[3][0] - contourExtremes[2][0]))
                        foamArea = np.sum(frame < 200) * pixelArea
                        frames.append(thresh)
                        areas.append(foamArea)
                else:
                    # print('\n\nVideo analysis completed!   ')
                    break
            time = np.linspace(0, 5, 51)
            areas.insert(0, 0)
            foamHeight.insert(0, 0)
            foamLength.insert(0, 0)
            df[flow][foam[:-1]] = pd.DataFrame({'time': time, 'area': areas,
                                                'length': foamLength, 'height': foamHeight})
    return df


def getCoeff():
    def f(x, m):
        return m*x
    ExpReg = {}
    ExpReg = {flow: {} for flow in flowrates}
    SimReg = {}
    SimReg = {flow: {} for flow in flowrates}
    Coeff = {}
    Coeff = {flow: {} for flow in flowrates}
    for flow in flowrates:
        for foam in foams:
            foam = foam[:-1]
            ExpReg[flow][foam] = linregress(dfExp[flow][foam]['time'], dfExp[flow][foam]['area'])
            SimReg[flow][foam] = linregress(dfSimCom[flow][foam]['time'],
                                            dfSimCom[flow][foam]['area'])
            z = SimReg[flow][foam].slope/ExpReg[flow][foam].slope
            Coeff[flow][foam] = [z, z*np.sqrt((SimReg[flow][foam].stderr/SimReg[flow][foam].slope)**2
                                              + (ExpReg[flow][foam].stderr/ExpReg[flow][foam].slope)
                                              ** 2)]
    return (Coeff, SimReg, ExpReg)


def averageCoeff(Coeff):
    CoeffFlow = {}
    CoeffFoam = {}
    for flow in flowrates:
        total = 0
        var = 0
        for foam in foams:
            foam = foam[:-1]
            total += Coeff[flow][foam][0]
            var += Coeff[flow][foam][1]**2
        ave = total/3
        std = np.sqrt(var)
        CoeffFlow[flow] = [ave, std]

    for foam in foams:
        foam = foam[:-1]
        total = 0
        var = 0
        for flow in flowrates:
            total += Coeff[flow][foam][0]
            var += Coeff[flow][foam][1]**2
        ave = total/3
        std = np.sqrt(var)
        CoeffFoam[foam] = [ave, std]

    total = 0
    var = 0
    for flow in flowrates:
        for foam in foams:
            foam = foam[:-1]
            total += Coeff[flow][foam][0]
            var += Coeff[flow][foam][1]**2
    ave = total/9
    std = np.sqrt(var)
    CoeffAve = [ave, std]
    return (CoeffAve, CoeffFlow, CoeffFoam)


def getCoeff2():
    def f(x, m):
        return m*x
    ExpReg = {}
    ExpReg = {flow: {} for flow in flowrates}
    SimReg = {}
    SimReg = {flow: {} for flow in flowrates}
    Coeff = {}
    Coeff = {flow: {} for flow in flowrates}
    for flow in flowrates:
        for foam in foams:
            foam = foam[:-1]
            ExpReg[flow][foam], covExp = curve_fit(f, dfExp[flow][foam]['time'],
                                                   dfExp[flow][foam]['area'])
            SimReg[flow][foam], covSim = curve_fit(f, dfSimCom[flow][foam]['time'],
                                                   dfSimCom[flow][foam]['area'])
            z = SimReg[flow][foam]/ExpReg[flow][foam]
            Coeff[flow][foam] = [z[0], z[0]*np.sqrt((covExp[0][0]/ExpReg[flow][foam][0])**2 +
                                                    (covSim[0][0]/SimReg[flow][foam][0])**2)]
    return (Coeff, SimReg, ExpReg)


def getBD(foam, flowrate):
    '''
    q, flowrate: flowrate in ml/min
    d: conduit diameter in m
    '''
    rheoData = {"DSS-1-3": {"K": 11.9293291220738, "n": 0.2819},
                "DSS-1-4": {"K": 8.14170811387433, "n": 0.4453},
                "DSS-1-5": {"K": 11.2683735402322, "n": 0.4062},
                "TSS-1-4": {"K": 10.2676687287686, "n": 0.3929},
                "TSS-1-5": {"K": 9.78646197990849, "n": 0.4237}}

    def flowToShear(q, d):
        return ((q*32)/(60000000*np.pi*(d/1000)**3))

    def getVisc(q, d, k, n):
        return (k*flowToShear(q, d)**(n-1))

    def getv0norm(phi):
        slope = 5.9149
        slopeErr = 0.2987
        intercept = -0.0887
        interceptErr = 0.0551
        v0norm = (slope*phi)+intercept
        err = np.sqrt((phi*slopeErr)**2+interceptErr**2)
        return (v0norm, err)

    def getKappa(phi):
        slope = 5.8871
        slopeErr = 1.0043
        intercept = -0.2732
        interceptErr = 0.1832
        kappa = (slope*phi)+intercept
        err = np.sqrt((phi*slopeErr)**2+interceptErr**2)
        return (kappa, err)
    q = int(flowrate[:2])
    mu = getVisc(q, 4.48, rheoData[foam]['K'], rheoData[foam]['n'])
    phi = 1/(int(foam[-1])+1)
    kappa = getKappa(phi)
    v0norm = getv0norm(phi)
    vm = q*5/60
    vg = vm * phi
    vcmcnorm = vg-((vg-v0norm[0])*np.exp(-kappa[0]*mu))
    vcmc = vcmcnorm * vg
    experr = np.exp(-kappa[0]*mu)*mu*kappa[1]
    err1 = vg*experr
    err2 = v0norm[0]*np.exp((-kappa[0]*mu))*np.sqrt((v0norm[1]/v0norm[0])**2 +
                                                    (experr/np.exp(-kappa[0]*mu))**2)
    finalErr = np.sqrt(err1**2+err2**2)
    return (vcmc, finalErr)


def getBDLine(foam, flowrate):
    '''
    q, flowrate: flowrate in ml/min
    d: conduit diameter in m
    '''
    rheoData = {"DSS-1-3": {"K": 11.9293291220738, "n": 0.2819},
                "DSS-1-4": {"K": 8.14170811387433, "n": 0.4453},
                "DSS-1-5": {"K": 11.2683735402322, "n": 0.4062},
                "TSS-1-4": {"K": 10.2676687287686, "n": 0.3929},
                "TSS-1-5": {"K": 9.78646197990849, "n": 0.4237}}

    def flowToShear(q, d):
        return ((q*32)/(60000000*np.pi*(d/1000)**3))

    def getVisc(q, d, k, n):
        return (k*flowToShear(q, d)**(n-1))

    def getv0norm(phi):
        slope = 5.9149
        intercept = -0.0887
        v0norm = (slope*phi)+intercept
        return v0norm

    def getKappa(phi):
        slope = 5.8871
        intercept = -0.2732
        kappa = (slope*phi)+intercept
        return kappa
    q = flowrate
    mu = getVisc(q, 4.48, rheoData[foam]['K'], rheoData[foam]['n'])
    phi = 1/(int(foam[-1])+1)
    kappa = getKappa(phi)
    v0norm = getv0norm(phi)
    vm = q*5/60
    vg = vm * phi
    vcmcnorm = vg-((vg-v0norm)*np.exp(-kappa*mu))
    vcmc = vcmcnorm * vg
    return vcmc


# %% Main
if __name__ == '__main__':
    figs = []
    titles = []
    g = getCoeff2
    dfSimIncomRf = getSimulationData(incompressible)
    dfSimComRf = getSimulationData(compressible)
    dfSimIncom = getArea(incompressible)
    dfSimCom = getArea(compressible)
    dfExp = getExperimentalData()
    # %%
    dfCoeff, dfSimReg, dfExpReg = g()
    dfCoeffAve, dfCoeffFlow, dfCoeffFoam = averageCoeff(dfCoeff)
    plt.style.use('ggplot')
    # %% Plot Separately
    # for flow in flowrates:
    #     fig, ax = plt.subplots(1, 3, figsize=(24, 12), constrained_layout=True)
    #     # fig.tight_layout()
    #     # fig.subplots_adjust(wspace=0.2, left=0.1, right=0.2, top=0.2)
    #     for i, foam in enumerate(foams):
    #         foam = foam[:-1]
    #         # fig, ax = plt.subplots(figsize=(12, 12))
    #         ax[i].scatter(dfSimIncomRf[flow][foam]['time'], dfSimIncomRf[flow][foam]['plane'],
    #                       c='r', s=60, marker='o', label='Sim Incom.')
    #         ax[i].scatter(dfSimComRf[flow][foam]['time'], dfSimComRf[flow][foam]['plane'],
    #                       edgecolors='g', s=60, marker='o', label='Sim Compr.', facecolors='none')
    #         ax[i].plot(dfExp[flow][foam]['time'], dfExp[flow][foam]['area'], label='Experiment')
    #         ax[i].set_title(f'{foam} {flow[:-1]}', pad=15, fontsize=18, position=[0.5, 1.2],
    #                         weight='bold')
    #         ax[i].grid(True, linestyle='-', linewidth=.5, which='minor')
    #         ax[i].grid(True, linestyle='-', linewidth=1.7, which='major')
    #         ax[i].tick_params(axis='both', labelsize=14)
    #         ax[i].set_xlim([0, 5])
    #         ax[i].set_ylim(([0, 250]))
    #         ax[i].set_xlabel('Time (s)', fontsize=18, labelpad=20)
    #         ax[i].set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
    #         ax[i].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #         ax[i].yaxis.set_minor_locator(plt.MultipleLocator(5))
    #         ax[i].legend(fontsize=14, loc='upper left', prop={'family': 'monospace', 'size': 18},
    #                      handletextpad=0.8)

    # %% Plot Compressible Report Files VS Compressible Projected Area
    fig, ax = plt.subplots(1, 3, figsize=(24, 12), constrained_layout=True)
    title = 'Compressible Report Files VS Compressible Projected Area'
    figs.append(fig)
    titles.append(title)
    fig.suptitle(title, fontsize=22,
                 fontweight='bold')
    for flow in flowrates:
        for i, foam in enumerate(foams):
            foam = foam[:-1]
            p = ax[i].scatter(dfSimComRf[flow][foam]['time'], dfSimComRf[flow][foam]['plane'],
                              s=60, marker='o', label=f'Rep. File {flow[:2]}' +
                              ' $\mathregular{ml.min^{-1}}$')
            ax[i].scatter(dfSimCom[flow][foam]['time'], dfSimCom[flow][foam]['area'],
                          s=60, marker='o', label=f'Pro. Area {flow[:2]}' +
                          ' $\mathregular{ml.min^{-1}}$', facecolors='none',
                          edgecolors=p.get_facecolor())
            ax[i].set_title(foam, pad=20, fontsize=18, position=[0.5, 1.2], weight='bold')
            ax[i].grid(True, linestyle='-', linewidth=.5, which='minor')
            ax[i].grid(True, linestyle='-', linewidth=1.7, which='major')
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].set_xlim([0, 5])
            ax[i].set_ylim(([0, 250]))
            ax[i].set_xlabel('Time (s)', fontsize=18, labelpad=20)
            ax[i].set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[i].yaxis.set_minor_locator(plt.MultipleLocator(5))
            ax[i].legend(loc='upper left', prop={'family': 'monospace', 'size': 14},
                         handletextpad=0.3)

    # %% Plot Incompressible Report Files VS Incompressible Projected Area
    fig, ax = plt.subplots(1, 3, figsize=(24, 12), constrained_layout=True)
    title = 'Incompressible Report Files VS Incompressible Projected Area'
    figs.append(fig)
    titles.append(title)
    fig.suptitle(title, fontsize=22,
                 fontweight='bold')
    for flow in flowrates:
        for i, foam in enumerate(foams):
            foam = foam[:-1]
            p = ax[i].scatter(dfSimIncomRf[flow][foam]['time'], dfSimIncomRf[flow][foam]['plane'],
                              s=60, marker='o', label=f'Rep. File {flow[:2]}' +
                              ' $\mathregular{ml.min^{-1}}$')
            ax[i].scatter(dfSimIncom[flow][foam]['time'], dfSimIncom[flow][foam]['area'], s=60,
                          marker='o', label=f'Pro. Area {flow[:2]}' + ' $\mathregular{ml.min^{-1}}$',
                          facecolors='none',
                          edgecolors=p.get_facecolor())
            ax[i].set_title(foam, pad=20, fontsize=18, position=[0.5, 1.2], weight='bold')
            ax[i].grid(True, linestyle='-', linewidth=.5, which='minor')
            ax[i].grid(True, linestyle='-', linewidth=1.7, which='major')
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].set_xlim([0, 5])
            ax[i].set_ylim(([0, 250]))
            ax[i].set_xlabel('Time (s)', fontsize=18, labelpad=20)
            ax[i].set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[i].yaxis.set_minor_locator(plt.MultipleLocator(5))
            ax[i].legend(loc='upper left', prop={'family': 'monospace', 'size': 14},
                         handletextpad=0.3)

    # %% Plot Compressible Projected Area VS Incompressible Projected Area
    fig, ax = plt.subplots(1, 3, figsize=(24, 12), constrained_layout=True)
    title = 'Compressible Projected Area VS Incompressible Projected Area'
    figs.append(fig)
    titles.append(title)
    fig.suptitle(title, fontsize=22,
                 fontweight='bold')
    for flow in flowrates:
        for i, foam in enumerate(foams):
            foam = foam[:-1]
            p = ax[i].scatter(dfSimIncom[flow][foam]['time'], dfSimIncom[flow][foam]['area'],
                              s=60, marker='o', label=f'Incom. {flow[:2]}' +
                              ' $\mathregular{ml.min^{-1}}$')
            ax[i].scatter(dfSimCom[flow][foam]['time'], dfSimCom[flow][foam]['area'],
                          s=60, marker='o', label=f'Compr. {flow[:2]}' +
                          ' $\mathregular{ml.min^{-1}}$', facecolors='none',
                          edgecolors=p.get_facecolor())
            ax[i].set_title(foam, pad=20, fontsize=18, position=[0.5, 1.2], weight='bold')
            ax[i].grid(True, linestyle='-', linewidth=.5, which='minor')
            ax[i].grid(True, linestyle='-', linewidth=1.7, which='major')
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].set_xlim([0, 5])
            ax[i].set_ylim(([0, 250]))
            ax[i].set_xlabel('Time (s)', fontsize=18, labelpad=20)
            ax[i].set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[i].yaxis.set_minor_locator(plt.MultipleLocator(5))
            ax[i].legend(loc='upper left', prop={'family': 'monospace', 'size': 14},
                         handletextpad=0.3)

    # %% Plot Compressible Projected Area VS Experimental
    fig, ax = plt.subplots(1, 3, figsize=(24, 12), constrained_layout=True)
    title = 'Compressible Projected Area VS Experimental'
    figs.append(fig)
    titles.append(title)
    fig.suptitle(title, fontsize=22, fontweight='bold')
    for flow in flowrates:
        for i, foam in enumerate(foams):
            foam = foam[:-1]
            ax[i].scatter(dfSimCom[flow][foam]['time'], dfSimCom[flow][foam]['area'],
                          s=60, marker='o', label=f'Simulation {flow[:2]}' +
                          ' $\mathregular{ml.min^{-1}}$')
            ax[i].plot(dfExp[flow][foam]['time'], dfExp[flow][foam]['area'],
                       label=f'Experiment {flow[:2]}' + ' $\mathregular{ml.min^{-1}}$')
            if g == getCoeff:
                ax[i].plot([0, 5], [x*dfSimReg[flow][foam].slope+dfSimReg[flow][foam].intercept
                                    for x in [0, 5]], linestyle='-.', color='g', zorder=6,
                           linewidth=0.5)
            else:
                ax[i].plot([0, 5], [x*dfSimReg[flow][foam] for x in [0, 5]],
                           linestyle='-.', color='g', zorder=6, linewidth=0.5)
            ax[i].set_title(foam, pad=20, fontsize=18, position=[0.5, 1.2], weight='bold')
            ax[i].grid(True, linestyle='-', linewidth=.5, which='minor')
            ax[i].grid(True, linestyle='-', linewidth=1.7, which='major')
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].set_xlim([0, 5])
            ax[i].set_ylim(([0, 250]))
            ax[i].set_xlabel('Time (s)', fontsize=18, labelpad=20)
            ax[i].set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[i].yaxis.set_minor_locator(plt.MultipleLocator(5))
            ax[i].legend(loc='upper left', prop={'family': 'monospace', 'size': 14},
                         handletextpad=0.3)

    # %% Bar Plot Correction Coefficient
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    title = 'Correction Coefficient 1'
    figs.append(fig)
    titles.append(title)
    fig.suptitle(title, fontsize=22, fontweight='bold')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    barWidth = 0.25
    r = np.arange(3)
    for i, flow in enumerate(flowrates):
        y = np.array(list(dfCoeff[flow].values()), dtype=float)[:, 0]
        err = np.array(list(dfCoeff[flow].values()), dtype=float)[:, 1]
        ax.bar(r, y, width=barWidth, yerr=err, ecolor=colors[i], capsize=5, label=f'{flow[:2]}' +
               ' $\mathregular{ml.min^{-1}}$')
        ax.set_ylim([0, 3.5])
        ax.legend(prop={'family': 'monospace', 'size': 16}, handletextpad=0.3)
        r = [x + barWidth for x in r]
    ax.set_xticks([r + barWidth for r in range(3)],
                  [x[4:].replace('-', ':') for x in list(dfCoeff[flow].keys())], fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(True, linestyle='-', linewidth=.7, which='minor')
    ax.grid(True, linestyle='-', linewidth=1.7, which='major')
    ax.set_xlabel('L:G', fontsize=18, labelpad=20)

    # %% Plot Averaged Correction Coefficient
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    title = 'Correction Coefficient 2'
    figs.append(fig)
    titles.append(title)
    fig.suptitle(title, fontsize=22, fontweight='bold')
    for flow, foam in zip(flowrates, foams):
        q = int(flow[:2])
        foam = foam[:-1]
        ax[0].set_title('Averaged Over Foam Formulation')
        ax[0].scatter(q, dfCoeffFlow[flow][0], s=40, c=colors[0])
        ax[0].errorbar(q, dfCoeffFlow[flow][0], yerr=dfCoeffFlow[flow][1], elinewidth=1.5,
                       ecolor=colors[0], capsize=5)
        ax[0].set_ylim([0, 3])
        ax[0].set_xlim([0, 12])
        ax[0].tick_params(axis='both', labelsize=14)
        ax[0].xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax[0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax[0].set_xlabel('Flowrate $\mathregular{ml.min^{-1}}$', fontsize=18, labelpad=20)
        ax[0].grid(True, linestyle='-', linewidth=.7, which='minor')
        ax[0].grid(True, linestyle='-', linewidth=1.7, which='major')
        ax[1].set_title('Averaged Over Flowrates')
        ax[1].scatter(foam, dfCoeffFoam[foam][0], s=40, c=colors[0])
        ax[1].errorbar(foam, dfCoeffFoam[foam][0], yerr=dfCoeffFoam[foam][1], elinewidth=1.5,
                       ecolor=colors[0], capsize=5)
        ax[1].set_ylim([0, 3])
        ax[1].set_xlim([-0.5, 2.5])
        ax[1].tick_params(axis='both', labelsize=14)
        ax[1].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax[1].set_xlabel('Foam Formulation', fontsize=18, labelpad=20)
        ax[1].grid(True, linestyle='-', linewidth=.7, which='minor')
        ax[1].grid(True, linestyle='-', linewidth=1.7, which='major')

    # %% Corrected Data
    fig, ax = plt.subplots(1, 3, figsize=(24, 12), constrained_layout=True)
    figs.append(fig)
    title = 'Overall Correction Factor'
    titles.append(title)
    fig.suptitle(title, fontsize=22, fontweight='bold')
    for j, flow in enumerate(flowrates):
        for i, foam in enumerate(foams):
            foam = foam[:-1]
            ax[i].plot(dfExp[flow][foam]['time'], dfExp[flow][foam]['area'],
                       label=f'{flow[:2]}' + ' $\mathregular{ml.min^{-1}}$')
            if g == getCoeff:
                ax[i].plot([0, 5], [x*(dfSimReg[flow][foam].slope/dfCoeffAve[0]) +
                                    (dfSimReg[flow][foam].intercept/dfCoeffAve[0])
                                    for x in [0, 5]],
                           linestyle='-.', color=colors[j], zorder=6, linewidth=0.5)
            else:
                ax[i].plot([0, 5], [x*dfSimReg[flow][foam]/dfCoeffAve[0] for x in [0, 5]],
                           linestyle='-.', color=colors[j], zorder=6, linewidth=0.5)
            ax[i].set_title(foam, pad=20, fontsize=18, position=[0.5, 1.2], weight='bold')
            ax[i].grid(True, linestyle='-', linewidth=.5, which='minor')
            ax[i].grid(True, linestyle='-', linewidth=1.7, which='major')
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].set_xlim([0, 5])
            ax[i].set_ylim(([0, 250]))
            ax[i].set_xlabel('Time (s)', fontsize=18, labelpad=20)
            ax[i].set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[i].yaxis.set_minor_locator(plt.MultipleLocator(5))
            ax[i].legend(loc='upper left', prop={'family': 'monospace', 'size': 18},
                         handletextpad=0.3, ncol=1)
    fig, ax = plt.subplots(1, 3, figsize=(24, 12), constrained_layout=True)
    figs.append(fig)
    title = 'Flowrate-Averaged Correction Factor'
    titles.append(title)
    fig.suptitle(title, fontsize=22, fontweight='bold')
    for j, flow in enumerate(flowrates):
        for i, foam in enumerate(foams):
            foam = foam[:-1]
            ax[i].plot(dfExp[flow][foam]['time'], dfExp[flow][foam]['area'],
                       label=f'{flow[:2]}' + ' $\mathregular{ml.min^{-1}}$')
            if g == getCoeff:
                ax[i].plot([0, 5], [x*(dfSimReg[flow][foam].slope/dfCoeffFlow[flow][0]) +
                                    (dfSimReg[flow][foam].intercept/dfCoeffFlow[flow][0])
                                    for x in [0, 5]],
                           linestyle='-.', color=colors[j], zorder=6, linewidth=0.5)
            else:
                ax[i].plot([0, 5], [x*dfSimReg[flow][foam]/dfCoeffFlow[flow][0] for x in [0, 5]],
                           linestyle='-.', color=colors[j], zorder=6, linewidth=0.5)
            ax[i].set_title(foam, pad=20, fontsize=18, position=[0.5, 1.2], weight='bold')
            ax[i].grid(True, linestyle='-', linewidth=.5, which='minor')
            ax[i].grid(True, linestyle='-', linewidth=1.7, which='major')
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].set_xlim([0, 5])
            ax[i].set_ylim(([0, 250]))
            ax[i].set_xlabel('Time (s)', fontsize=18, labelpad=20)
            ax[i].set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[i].yaxis.set_minor_locator(plt.MultipleLocator(5))
            ax[i].legend(loc='upper left', prop={'family': 'monospace', 'size': 18},
                         handletextpad=0.3, ncol=1)
    fig, ax = plt.subplots(1, 3, figsize=(24, 12), constrained_layout=True)
    figs.append(fig)
    title = 'Foam-Averaged Correction Factor'
    titles.append(title)
    fig.suptitle(title, fontsize=22, fontweight='bold')
    for j, flow in enumerate(flowrates):
        for i, foam in enumerate(foams):
            foam = foam[:-1]
            ax[i].plot(dfExp[flow][foam]['time'], dfExp[flow][foam]['area'],
                       label=f'{flow[:2]}' + ' $\mathregular{ml.min^{-1}}$')
            if g == getCoeff:
                ax[i].plot([0, 5], [x*(dfSimReg[flow][foam].slope/dfCoeffFoam[foam][0]) +
                                    (dfSimReg[flow][foam].intercept/dfCoeffFoam[foam][0])
                                    for x in [0, 5]],
                           linestyle='-.', color=colors[j], zorder=6, linewidth=0.5)
            else:
                ax[i].plot([0, 5], [x*dfSimReg[flow][foam]/dfCoeffFoam[foam][0] for x in [0, 5]],
                           linestyle='-.', color=colors[j], zorder=6, linewidth=0.5)
            ax[i].set_title(foam, pad=20, fontsize=18, position=[0.5, 1.2], weight='bold')
            ax[i].grid(True, linestyle='-', linewidth=.5, which='minor')
            ax[i].grid(True, linestyle='-', linewidth=1.7, which='major')
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].set_xlim([0, 5])
            ax[i].set_ylim(([0, 250]))
            ax[i].set_xlabel('Time (s)', fontsize=18, labelpad=20)
            ax[i].set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[i].yaxis.set_minor_locator(plt.MultipleLocator(5))
            ax[i].legend(loc='upper left', prop={'family': 'monospace', 'size': 18},
                         handletextpad=0.3, ncol=1)

    # %% BD Comparison
    fig, ax = plt.subplots(1, 3, figsize=(24, 12), constrained_layout=True)
    figs.append(fig)
    title = 'Comparison of Displaced CMC'
    titles.append(title)
    fig.suptitle(title, fontsize=22, fontweight='bold')
    for i, foam in enumerate(foams):
        foam = foam[:-1]
        tempSim = []
        tempPred = []
        x = []
        for j, flow in enumerate(flowrates):
            q = int(flow[:2])
            simBD = abs(dfSimComRf[flow][foam]['bd-front'][50]*5/60) + abs(dfSimComRf[flow][foam]['bd-back'][50]*5/60)
            predBD = getBD(foam, flow)
            x.append(q)
            tempSim.append(simBD)
            tempPred.append(predBD[0])
            ax[i].scatter(q, simBD, label='Simulated', color='k', marker='s', s=100)
            # ax[i].scatter(q, simBD/dfCoeffFoam[foam][0], label='Corrected', color='k',
            #               marker='X', s=100)
            ax[i].scatter(q, predBD[0], color='k', label='Empirical', s=100)
            ax[i].errorbar(q, predBD[0], yerr=predBD[1], elinewidth=1.5, ecolor='k', capsize=5)
            ax[i].set_title(foam, pad=20, fontsize=18, position=[0.5, 1.2], weight='bold')
            ax[i].set_xlim([0, 12])
            ax[i].set_ylim(([-0.1, 1]))
            ax[i].set_xlabel('Flowrate $\mathregular{ml.min^{-1}}$', fontsize=18, labelpad=20)
            ax[i].set_ylabel('$\mathregular{V_{CMC}}$ (mL)', fontsize=18, labelpad=15)
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(1))
            ax[i].yaxis.set_minor_locator(plt.MultipleLocator(0.05))
            ax[i].grid(True, linestyle='-', linewidth=.5, which='minor')
            ax[i].grid(True, linestyle='-', linewidth=1.7, which='major')
            if j == 0:
                ax[i].legend(prop={'family': 'monospace', 'size': 18}, handletextpad=0.3)
        simReg = linregress(x, tempSim)
        predReg = linregress(x, tempPred)
        ax[i].plot([0, 12],[0, simReg.slope*12+simReg.intercept],
                   color='k', linestyle='-.', linewidth=0.5)
        xLine = np.linspace(1e-5,12,100)
        y = []
        for f in xLine:
            y.append(getBDLine(foam, f))
        ax[i].plot(xLine, y, color='k', linestyle='-.', linewidth=0.5)
    # %% Print Data Tail
    for flow in flowrates:
        for foam in foams:
            foam = foam[:-1]
            print()
            print(foam)
            print(flow)
            print('\nExperimental Data')
            print(dfExp[flow][foam][['time', 'area']].tail())
            print('\nCompressible Report Files')
            print(dfSimComRf[flow][foam][['time', 'plane']].tail())
            print('\nCompressible Projected Area')
            print(dfSimCom[flow][foam][['time', 'area']].tail())
            print('\nIncompressible Report Files')
            print(dfSimIncomRf[flow][foam][['time', 'plane']].tail())
            print('\nIncompressible Projected Area')
            print(dfSimIncom[flow][foam][['time', 'area']].tail())
            break
        break

    # %% Print Data Tail
    ui = input('Save Figures?\n>> ')
    if ui == '':
        for fig, title in zip(figs, titles):
            fig.savefig(home+plots+title+'.png', dpi=450, transparent=False)
