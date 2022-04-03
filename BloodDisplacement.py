#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:44:40 2021

@author: alizz
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import pearsonr


# %% Functions

def visc(data, vdata):
    mu = {}
    nerr = {}
    kerr = {}
    muerr = {}
    for tech in data.keys():
        mu[tech] = {}
        nerr[tech] = {}
        kerr[tech] = {}
        muerr[tech] = {}
        for lg in data[tech].keys():
            mu[tech][lg] = {}

            nerr[tech][lg] = vdata[tech][lg]['nerror']
            kerr[tech][lg] = vdata[tech][lg]['kerror']
            for flowrate in data[tech][lg]['x']:
                mu[tech][lg][str(flowrate)] = {}
                gamma = flowToShear(flowrate, 4.48/1000)
                mu[tech][lg][str(flowrate)]['viscosity'] = flowToVisc(flowrate, 4.48/1000, 
                                                                      vdata[tech][lg]['k'],
                                                                      vdata[tech][lg]['n'])
                z = np.exp((vdata[tech][lg]['n']-1)*np.log(gamma))
                zerr = z*vdata[tech][lg]['nerror']*np.log(gamma)
                mu[tech][lg][str(flowrate)]['viscosityError'] = mu[tech][lg][str(flowrate)]['viscosity']*np.sqrt(
                    ((vdata[tech][lg]['kerror']**2)/(vdata[tech][lg]['k']**2))+((zerr**2)/(z**2)))
    return (mu)


def pLaw(shearRate, k, n):
    tw = (k*(shearRate**n))
    return tw


def apparentVisc(shearRate, k, n):
    vis = (k*(shearRate**(n-1)))
    return vis


def flowToShear(x, d):
    return ((x*32)/(60000000*np.pi*d**3))


def shearToFlow(x, d):
    return ((x*60000000*np.pi*d**3)/32)


def flowToVisc(flow, d, k, n):
    shear = flowToShear(flow, d)
    return apparentVisc(shear, k, n)


def plateau(x, k, y0):
    y = 7-(7-y0)*np.exp(-k*x)
    return y


def plateau3(x, k, y0):
    a = 7/4
    y = a-(a-y0)*np.exp(-k*x)
    return y


def plateau4(x, k, y0):
    a = 7/5
    y = a-(a-y0)*np.exp(-k*x)
    return y


def plateau5(x, k, y0):
    a = 7/6
    y = a-(a-y0)*np.exp(-k*x)
    return y


def plateauDR(x, k, y0):
    y = 1-(1-y0)*np.exp(-k*x)
    return y


def isoline(x, k, y0, x0):
    y = y0+np.log(k*(-x+x0))
    return y


def lin(x, m, b):
    y = m*x+b
    return y


def expo(x, a, m, b):
    y = [a*np.exp(xx*m)+b for xx in x]
    return y

# %% Read BD CSVs


homeBd = '/Users/alizz/My Drive (alireza.meghdadi@gmail.com)/Southampton - GoogleDrive/Experiments/'
homeBd += 'Blood Displacement/Data/'

directory = os.fsencode(homeBd)
directory_files = sorted(os.listdir(directory))
rhoCmc = 0.98459  # g/ml
foamVolume = 7  # mL
plotSize = [12, 12]
csvs = []
cmcdata = {}

for file in directory_files:
    filename = os.fsdecode(file)
    if filename.endswith('.csv'):
        csvs.append(filename)
        cmcdata[filename[:-4]] = {}
        with open(homeBd+filename) as csvfile:
            reader = csv.reader(csvfile)
            rownum = 1
            for row in reader:
                if rownum == 1:
                    flowrates = row[1:]
                    cmcdata[filename[:-4]][flowrates[0]] = []
                    cmcdata[filename[:-4]][flowrates[1]] = []
                    cmcdata[filename[:-4]][flowrates[2]] = []
                    cmcdata[filename[:-4]][flowrates[3]] = []
                    rownum += 1
                else:
                    cmcdata[filename[:-4]][flowrates[0]].append(row[1])
                    cmcdata[filename[:-4]][flowrates[1]].append(row[2])
                    cmcdata[filename[:-4]][flowrates[2]].append(row[3])
                    cmcdata[filename[:-4]][flowrates[3]].append(row[4])
                    rownum += 1

# %% Read Power-law CSVs
homePl = '/Users/alizz/My Drive (alireza.meghdadi@gmail.com)/Southampton - GoogleDrive/Experiments/Pipe Viscometry/Data/Processed Results/'

directory = os.fsencode(homePl)
directory_files = sorted(os.listdir(directory))

viscdata = {}
for file in directory_files:
    filename = os.fsdecode(file)
    if filename.startswith('Power') and filename.endswith('.csv'):
        with open(homePl+filename) as csvfile:
            reader = csv.reader(csvfile)
            viscdata['DSS'] = {}
            viscdata['TSS'] = {}

            for row in reader:
                if row[0].startswith('4.48'):
                    viscdata[row[1]][row[2]] = {'n'         : float(row[3]),
                                                'nerror'    : float(row[4]),
                                                'k'         : float(row[9]),
                                                'kerror'    : float(row[10])}
                else:
                    continue

# %% Calculate Blood Displacement
displacedVol = {}
for tech in ['DSS', 'TSS']:
    displacedVol[tech] = {}
    for lg in ['1-3', '1-4', '1-5']:
        displacedVol[tech][lg] = {}

for foam in cmcdata.keys():
    for flowrate in cmcdata[foam].keys():
        i = 0
        displacedVol[foam[:3]][foam[6:]][flowrate] = []
        while i < 20:
            displacedVol[foam[:3]][foam[6:]][flowrate].append((float(cmcdata[foam][flowrate][i+1]) - float(cmcdata[foam][flowrate][i]))/(1000*rhoCmc))
            i += 2

displacement = {}
displacementRatio = {}
for tech in displacedVol.keys():
    displacement[tech] = {}
    displacementRatio[tech] = {}
    for lg in displacedVol[tech].keys():
        displacement[tech][lg] = {}
        displacement[tech][lg]['x'] = []
        displacement[tech][lg]['y'] = []
        displacement[tech][lg]['err'] = []
        displacementRatio[tech][lg] = {}
        displacementRatio[tech][lg]['x'] = []
        displacementRatio[tech][lg]['y'] = []
        displacementRatio[tech][lg]['err'] = []
        for flowrate in displacedVol[tech][lg].keys():
            displacement[tech][lg]['x'].append(int(flowrate))
            displacement[tech][lg]['y'].append(np.average(displacedVol[tech][lg][flowrate]))
            displacement[tech][lg]['err'].append(np.std(displacedVol[tech][lg][flowrate]))
            displacementRatio[tech][lg]['x'].append(int(flowrate))
            displacementRatio[tech][lg]['y'].append(np.average(displacedVol[tech][lg][flowrate])/foamVolume)
            displacementRatio[tech][lg]['err'].append(np.std(displacedVol[tech][lg][flowrate])/foamVolume)

# %% Plot Graphs

plt.style.use('ggplot')
figs = []
markers = ['o', 's', '^']

# %% Plot Master CMC Displaced

colours = []
mu = visc(displacement,viscdata)
figs.append(plt.figure(figsize=plotSize))
figs[-1].subplots_adjust(wspace=0.25, hspace=0.5)
figs[-1].suptitle('Master CMC Displaced',weight='bold',fontsize=30)

xdata = {}
ydata = {}
yerrdata = {}
for key in ['1-3','1-4','1-5']:
    xdata[key] = []
    ydata[key] = []
    yerrdata[key] = []

j = 0
for lg in ['1-3','1-4','1-5']:
    i = 0
    for tech in ['DSS', 'TSS']:
        x = [mu[tech][lg][x]['viscosity'] for x in flowrates]
        y = displacement[tech][lg]['y']
        yerr = displacement[tech][lg]['err']
        xerr = [mu[tech][lg][x]['viscosityError'] for x in flowrates]
        xdata[lg] += x
        ydata[lg] += y
        yerrdata[lg] += yerr
        labs = ['Q = %s $\mathregular{ml.{min}^{-1}}$'%x.zfill(2) for x in flowrates]
        if i == 0:
            plt.scatter(x, y, marker=markers[j], label = tech + ' | '+ lg.replace('-',':'),linewidth=1,s=60)
            g = plt.plot(x,y,linestyle='')
            colours.append(g[-1].get_color())
        else:
            plt.scatter(x, y, marker=markers[j], facecolor = plt.gca().get_facecolor(), color = colours[-1], linewidth=1,label = tech + ' | '+ lg.replace('-',':'),zorder = 6 ,s=60)
        ax = plt.gca()

        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.minorticks_on()
        ax.grid(True, linestyle=':',linewidth=1.5,which='minor')
        ax.grid(True, linestyle='-',linewidth=1.5,which='major')
        plt.ylim([5.8,7.2])
        plt.xlim([0,4])
        plt.tick_params(axis='both', labelsize=18)
        plt.tick_params(axis='both', which = 'major', length=7, width=1.7)
        plt.tick_params(axis='both', which = 'minor', length=4, width=1)
        plt.xlabel('Foam $\mathregular{{\mu}_{App}}$ (Pa.s)', fontsize = 20, labelpad = 20)
        plt.ylabel('$\mathregular{V}_{CMC}$ (mL)', fontsize = 20, labelpad = 15)
        plt.errorbar(x, y, yerr=yerr, xerr=xerr,fmt='None',ecolor=colours[-1],elinewidth=1.5,capsize=4,zorder=4)

        i += 1
    j += 1

leg1 = ax.legend(ncol=3, prop={'family': 'monospace', 'size': 18}, loc='upper left', handletextpad = 0)
x1 = {}
y1 = {}
yerr1 = {}
alpha = [1,0.6,0.45,0.3]
bound = ([2,2,2],[4.9,4.9,4.9])
isos = []
isoPar = {}
parameters = {}
pcov = {}
for i in range(0,4):
    x1[labs[i][4:6]] = []
    y1[labs[i][4:6]] = []
    yerr1[labs[i][4:6]] = []
    isoPar[labs[i][4:6]] = []
    for lg in ['1-3','1-4','1-5']:
        x1[labs[i][4:6]].append(xdata[lg][i])
        x1[labs[i][4:6]].append(xdata[lg][i+4])
        y1[labs[i][4:6]].append(ydata[lg][i])
        y1[labs[i][4:6]].append(ydata[lg][i+4])
        yerr1[labs[i][4:6]].append(yerrdata[lg][i])
        yerr1[labs[i][4:6]].append(yerrdata[lg][i+4])

    xlin = np.linspace(0,10,1000)
    ylin = np.linspace(0,10,1000)
    params = curve_fit(isoline,x1[labs[i][4:6]],y1[labs[i][4:6]],sigma=yerr1[labs[i][4:6]],bounds = bound)[0]
    isoPar[labs[i][4:6]] = params
    isos += (plt.plot(xlin,isoline(xlin,*params), color = 'k', linewidth=1.2, linestyle=(0,(3,5,1,5,1,5)), label = labs[i],alpha=alpha[i]))
leg2 = ax.legend(isos, labs, prop={'family': 'monospace', 'size': 18}, loc='lower right')
ax.add_artist(leg1)

i = 0
for lg in xdata.keys():
    xlin = np.linspace(0,10,1000)
    parameters[lg], pcov[lg] = curve_fit(plateau,xdata[lg],ydata[lg],sigma=yerrdata[lg])
    plt.plot(xlin,plateau(xlin,*parameters[lg]), color=colours[i], linewidth=1.2, linestyle='-.', label = '')
    i += 1

# %% Plot Master Displacement Ratio

colours = []
figs.append(plt.figure(figsize=plotSize))
figs[-1].subplots_adjust(wspace=0.25, hspace=0.5)
figs[-1].suptitle('Master Displacement Ratio',weight='bold',fontsize=30)

xdata = {}
ydata = {}
yerrdata = {}
for key in ['1-3','1-4','1-5']:
    xdata[key] = []
    ydata[key] = []
    yerrdata[key] = []

j = 0
for lg in ['1-3','1-4','1-5']:
    i = 0
    for tech in ['DSS', 'TSS']:
        x = [mu[tech][lg][x]['viscosity'] for x in flowrates]
        y = displacementRatio[tech][lg]['y']
        yerr = displacementRatio[tech][lg]['err']
        xerr = [mu[tech][lg][x]['viscosityError'] for x in flowrates]
        xdata[lg] += x
        ydata[lg] += y
        yerrdata[lg] += yerr
        labs = ['Q = %s $\mathregular{ml.{min}^{-1}}$'%x.zfill(2) for x in flowrates]
        if i == 0:
            plt.scatter(x, y, marker=markers[j], label = tech + ' | '+ lg.replace('-',':'),linewidth=1, s=60)
            g = plt.plot(x,y,linestyle='')
            colours.append(g[-1].get_color())
        else:
            plt.scatter(x, y, marker=markers[j], facecolor = plt.gca().get_facecolor(), color = colours[-1], linewidth=1,label = tech + ' | '+ lg.replace('-',':'),zorder = 6, s=60)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.minorticks_on()
        ax.grid(True, linestyle=':',linewidth=1.5,which='minor')
        ax.grid(True, linestyle='-',linewidth=1.5,which='major')
        plt.ylim([0.8,1.05])
        plt.xlim([0,4])
        plt.tick_params(axis='both', labelsize=18)
        plt.tick_params(axis='both', which = 'major', length=7, width=1.7)
        plt.tick_params(axis='both', which = 'minor', length=4, width=1)
        plt.xlabel('$\mathregular{{\mu}_{App}}$ (Pa.s)', fontsize = 20, labelpad = 20)
        plt.ylabel('Displacement Ratio', fontsize = 20, labelpad = 15)
        plt.errorbar(x, y, yerr=yerr, xerr=xerr,fmt='None',ecolor=colours[-1],elinewidth=1.5,capsize=4,zorder=4)
        i += 1
    j += 1

leg1 = ax.legend(ncol=3, prop={'family': 'monospace', 'size': 18}, loc='upper left', handletextpad = 0)

x1 = {}
y1 = {}
yerr1 = {}
bound = (3,5)
isos = []

for i in range(0,4):
    x1[labs[i][4:6]] = []
    y1[labs[i][4:6]] = []
    yerr1[labs[i][4:6]] = []
    for lg in ['1-3','1-4','1-5']:
        x1[labs[i][4:6]].append(xdata[lg][i])
        x1[labs[i][4:6]].append(xdata[lg][i+4])
        y1[labs[i][4:6]].append(ydata[lg][i])
        y1[labs[i][4:6]].append(ydata[lg][i+4])
        yerr1[labs[i][4:6]].append(yerrdata[lg][i])
        yerr1[labs[i][4:6]].append(yerrdata[lg][i+4])

    xlin = np.linspace(0,10,1000)
    ylin = np.linspace(0,1,1000)
    isos += (plt.plot(xlin,[x/foamVolume for x in isoline(xlin,*isoPar[labs[i][4:6]])], color = 'k', linewidth=1.2, linestyle=(0,(3,5,1,5,1,5)), label = labs[i],alpha=alpha[i]))
leg2 = ax.legend(isos, labs, prop={'family': 'monospace', 'size': 18}, loc='lower right')
ax.add_artist(leg1)

i = 0
for lg in xdata.keys():
    xlin = np.linspace(0,10,1000)
    params = curve_fit(plateauDR,xdata[lg],ydata[lg],sigma=yerrdata[lg])[0]
    plt.plot(xlin,plateauDR(xlin,*params), color=colours[i], linewidth=1.2, linestyle='-.')
    i += 1

# %% Plot Master CMC Displaced per Unit Volume Gas

colours = []
figs.append(plt.figure(figsize=plotSize))
figs[-1].subplots_adjust(wspace=0.25, hspace=0.5)
figs[-1].suptitle('Master CMC Displaced per Unit Volume Gas',weight='bold',fontsize=30)


xdata = {}
ydata = {}
parameters2 ={}
pcov2 = {}
yerrdata = {}
for key in ['1-3','1-4','1-5']:
    xdata[key] = []
    ydata[key] = []
    yerrdata[key] = []

j = 0
for lg in ['1-3','1-4','1-5']:
    i = 0
    for tech in ['DSS', 'TSS']:
        x = [mu[tech][lg][x]['viscosity'] for x in flowrates]
        y = [x*(1/(float(lg[-1])+1)) for x in displacement[tech][lg]['y']]
        yerr = [x*(1/(float(lg[-1])+1)) for x in displacement[tech][lg]['err']]
        xerr = [mu[tech][lg][x]['viscosityError'] for x in flowrates]
        xdata[lg] += x
        ydata[lg] += y
        yerrdata[lg] += yerr
        labs = ['Q = %s $\mathregular{ml.{min}^{-1}}$'%x.zfill(2) for x in flowrates]
        if i == 0:
            plt.scatter(x, y, marker=markers[j], label = tech + ' | '+ lg.replace('-',':'),linewidth=1,s=60)
            g = plt.plot(x,y,linestyle='')
            colours.append(g[-1].get_color())
        else:
            plt.scatter(x, y, marker=markers[j], facecolor = plt.gca().get_facecolor(), color = colours[-1], linewidth=1,label = tech + ' | '+ lg.replace('-',':'),zorder = 6, s=60)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.minorticks_on()
        ax.grid(True, linestyle=':',linewidth=1.5,which='minor')
        ax.grid(True, linestyle='-',linewidth=1.7,which='major')
        plt.ylim([0.8,1.8])
        plt.xlim([0,4])
        plt.tick_params(axis='both', labelsize=18)
        plt.tick_params(axis='both', which = 'major', length=7, width=1.7)
        plt.tick_params(axis='both', which = 'minor', length=4, width=1)
        plt.xlabel('$\mathregular{{\mu}_{App}}$ (Pa.s)', fontsize = 20, labelpad = 20)
        plt.ylabel('$\mathregular{V}_{CMC}/{V}_{g}$', fontsize = 20, labelpad = 15)
        plt.legend(ncol=3, prop={'family': 'monospace', 'size': 18}, loc='lower left', handletextpad = 0)
        plt.errorbar(x, y, yerr=yerr, xerr=xerr,fmt='None',ecolor=colours[-1],elinewidth=1.5,capsize=4,zorder=4)
        i += 1
    j += 1

i = 0
functions = [plateau3,plateau4,plateau5]
for lg in xdata.keys():
    xlin = np.linspace(0,10,1000)
    parameters2[lg], pcov2[lg] = curve_fit(functions[i],xdata[lg],ydata[lg],sigma=yerrdata[lg])
    plt.plot(xlin,functions[i](xlin,*parameters2[lg]), color=colours[i], linewidth=1.2, linestyle='-.')
    i += 1

# %% K VS L:G
ticklabelsize = 24
axislabelsize = 28

results ={}
plateauResults = {}
figs.append(plt.figure(figsize=plotSize))
figs[-1].subplots_adjust(wspace=0.25, hspace=0.5)
figs[-1].suptitle('CMC Volume - Curve Parameters K',weight='bold',fontsize=30)

x = []
y = []
std = []
for lg in parameters.keys():
    x.append(1/(float(lg[-1])+1))
    y.append(parameters[lg][0])
    std.append(np.sqrt(np.diag(pcov[lg]))[0]/2)

key = 'K'
plateauResults[key] = {}

results[key] = curve_fit(lin,x,y, sigma=std,absolute_sigma=True)
zslope = (results[key][0][0]/np.sqrt(np.diag(results[key][1]))[0])
pslope = norm.sf(zslope)
rsq = pearsonr(x,y)[0]**2
plateauResults[key]['x'] = x
plateauResults[key]['y'] = y
plateauResults[key]['std'] = std
plateauResults[key]['r2'] = rsq
plateauResults[key]['pearson-p'] = pearsonr(x,y)[1]
plateauResults[key]['slope-p'] = pslope

label = "$\mathregular{y={%.4f}x%.4f}$, $R^{2}=%.4f$\n$\mathregular{\sigma}_{slope}$=%0.4f, $\mathregular{\sigma}_{intercept}$=%0.4f"%(results[key][0][0],results[key][0][1],rsq, np.sqrt(np.diag(results[key][1]))[0],np.sqrt(np.diag(results[key][1]))[1])

plt.scatter(x,y)
plt.errorbar(x,y, yerr=std,elinewidth=1.5,capsize=4, linestyle='none')
plt.plot([0.05,0.35],[results[key][0][0]*x+results[key][0][1] for x in [0.05,0.35]], linestyle='-.',linewidth=1.2, color='g',label = label)
ax = plt.gca()
plt.minorticks_on()
ax.grid(True, linestyle=':',linewidth=1.5,which='minor')
ax.grid(True, linestyle='-',linewidth=1.7,which='major')
plt.ylim([0,1.8])
plt.xlim([0,0.4])
plt.tick_params(axis='both', labelsize=ticklabelsize, pad = 15)
plt.tick_params(axis='both', which = 'major', length=7, width=1.7)
plt.tick_params(axis='both', which = 'minor', length=4, width=1)
plt.xlabel('$\mathit{{\phi}_{g}}$', fontsize = axislabelsize, labelpad = 16)
plt.ylabel('$\mathregular{\kappa}$ $\mathregular{(Pa.s)^{-1}}$', fontsize = axislabelsize, labelpad = 12)
plt.legend(ncol=1, prop={'family': 'monospace', 'size': ticklabelsize}, handletextpad = 0.8)

results2 ={}
plateauResults2 = {}
figs.append(plt.figure(figsize=plotSize))
figs[-1].subplots_adjust(wspace=0.25, hspace=0.5)
figs[-1].suptitle('CMC Volume Per Gas - Curve Parameters K',weight='bold',fontsize=30)

x = []
y = []
std = []
for lg in parameters.keys():
    x.append(1/(float(lg[-1])+1))
    y.append(parameters2[lg][0])
    std.append(np.sqrt(np.diag(pcov2[lg]))[0]/2)

key = 'K'
plateauResults2[key] = {}

results2[key] = curve_fit(lin,x,y, sigma=std,absolute_sigma=True)
zslope = (results2[key][0][0]/np.sqrt(np.diag(results2[key][1]))[0])
pslope = norm.sf(zslope)
rsq = pearsonr(x,y)[0]**2
plateauResults2[key]['x'] = x
plateauResults2[key]['y'] = y
plateauResults2[key]['std'] = std
plateauResults2[key]['r2'] = rsq
plateauResults2[key]['pearson-p'] = pearsonr(x,y)[1]
plateauResults2[key]['slope-p'] = pslope
label = "$\mathregular{y={%.4f}x%.4f}$, $R^{2}=%.4f$\n$\mathregular{\sigma}_{slope}$=%0.4f, $\mathregular{\sigma}_{intercept}$=%0.4f"%(results2[key][0][0],results2[key][0][1],rsq, np.sqrt(np.diag(results2[key][1]))[0],np.sqrt(np.diag(results2[key][1]))[1])

plt.scatter(x,y)
plt.errorbar(x,y, yerr=std,elinewidth=1.5,capsize=4, linestyle='none')
plt.plot([0.05,0.35],[results2[key][0][0]*x+results2[key][0][1] for x in [0.05,0.35]], linestyle='-.',linewidth=1.2, color='g',label = label)
ax = plt.gca()
plt.minorticks_on()
ax.grid(True, linestyle=':',linewidth=1.5,which='minor')
ax.grid(True, linestyle='-',linewidth=1.7,which='major')
plt.ylim([0,1.8])
plt.xlim([0,0.4])
plt.tick_params(axis='both', labelsize=ticklabelsize, pad = 15)
plt.tick_params(axis='both', which = 'major', length=7, width=1.7)
plt.tick_params(axis='both', which = 'minor', length=4, width=1)
plt.xlabel('$\mathit{{\phi}_{g}}$', fontsize = axislabelsize, labelpad =16)
plt.ylabel('$\mathregular{\kappa}$ $\mathregular{(Pa.s)^{-1}}$', fontsize = axislabelsize, labelpad = 12)
plt.legend(ncol=1, prop={'family': 'monospace', 'size': ticklabelsize}, handletextpad = 0.8)

# %% Y0 VS L:G

figs.append(plt.figure(figsize=plotSize))
figs[-1].subplots_adjust(wspace=0.25, hspace=0.5)
figs[-1].suptitle('CMC Volume - Curve Parameters V0',weight='bold',fontsize=30)

x = []
y = []
std = []
for lg in parameters.keys():
    x.append(1/(float(lg[-1])+1))
    y.append(parameters[lg][1])
    std.append(np.sqrt(np.diag(pcov[lg]))[1]/2)

key = 'Y0'
plateauResults[key] = {}

results[key] = curve_fit(lin,x,y, sigma = std, absolute_sigma=True)
zslope = (results[key][0][0]/np.sqrt(np.diag(results[key][1]))[0])
pslope = norm.sf(zslope)
rsq = pearsonr(x,y)[0]**2
plateauResults[key]['x'] = x
plateauResults[key]['y'] = y
plateauResults[key]['std'] = std
plateauResults[key]['r2'] = rsq
plateauResults[key]['pearson-p'] = pearsonr(x,y)[1]
plateauResults[key]['slope-p'] = pslope
label = "$\mathregular{y={%.4f}x+%.4f}$, $R^{2}=%.4f$\n$\mathregular{\sigma}_{slope}$=%0.4f, $\mathregular{\sigma}_{intercept}$=%0.4f"%(results[key][0][0],results[key][0][1], rsq, np.sqrt(np.diag(results[key][1]))[0],np.sqrt(np.diag(results[key][1]))[1])

plt.scatter(x,y)
plt.errorbar(x,y, yerr=std,elinewidth=1.5,capsize=4, linestyle='none')
plt.plot([0.05,0.35],[results[key][0][0]*x+results[key][0][1] for x in [0.05,0.35]], linestyle='-.',linewidth=1.2, color='g',label = label)
ax = plt.gca()
plt.minorticks_on()
ax.grid(True, linestyle=':',linewidth=1.5,which='minor')
ax.grid(True, linestyle='-',linewidth=1.7,which='major')
plt.ylim([5,6])
plt.xlim([0,0.4])
plt.tick_params(axis='both', labelsize=ticklabelsize, pad = 15)
plt.tick_params(axis='both', which = 'major', length=7, width=1.7)
plt.tick_params(axis='both', which = 'minor', length=4, width=1)
plt.xlabel('$\mathit{{\phi}_{g}}$', fontsize = axislabelsize, labelpad = 16)
plt.ylabel('$\mathregular{{V}_{0}}$ (mL)', fontsize = axislabelsize, labelpad = 12)
plt.legend(ncol=1, prop={'family': 'monospace', 'size': ticklabelsize}, handletextpad = 0.8, labelspacing=1)

figs.append(plt.figure(figsize=plotSize))
figs[-1].subplots_adjust(wspace=0.25, hspace=0.5)
figs[-1].suptitle('CMC Volume Per Gas - Curve Parameters V0',weight='bold',fontsize=30)

x = []
y = []
std = []
for lg in parameters.keys():
    x.append(1/(float(lg[-1])+1))
    y.append(parameters2[lg][1])
    std.append(np.sqrt(np.diag(pcov2[lg]))[1]/2)

key = 'Y0'
plateauResults2[key] = {}

results2[key] = curve_fit(lin,x,y, sigma = std, absolute_sigma=True)
zslope = (results2[key][0][0]/np.sqrt(np.diag(results2[key][1]))[0])
pslope = norm.sf(zslope)
rsq = pearsonr(x,y)[0]**2
plateauResults2[key]['x'] = x
plateauResults2[key]['y'] = y
plateauResults2[key]['std'] = std
plateauResults2[key]['r2'] = rsq
plateauResults2[key]['pearson-p'] = pearsonr(x,y)[1]
plateauResults2[key]['slope-p'] = pslope
label = "$\mathregular{y={%.4f}x%.4f}$, $R^{2}=%.4f$\n$\mathregular{\sigma}_{slope}$=%0.4f, $\mathregular{\sigma}_{intercept}$=%0.4f"%(results2[key][0][0],results2[key][0][1], rsq, np.sqrt(np.diag(results2[key][1]))[0],np.sqrt(np.diag(results2[key][1]))[1])

plt.scatter(x,y)
plt.errorbar(x,y, yerr=std,elinewidth=1.5,capsize=4, linestyle='none')
plt.plot([0.05,0.35],[results2[key][0][0]*x+results2[key][0][1] for x in [0.05,0.35]], linestyle='-.',linewidth=1.2, color='g',label = label)
ax = plt.gca()
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.minorticks_on()
ax.grid(True, linestyle=':',linewidth=1.5,which='minor')
ax.grid(True, linestyle='-',linewidth=1.7,which='major')
plt.ylim([0,2])
plt.xlim([0,0.4])
plt.tick_params(axis='both', labelsize=ticklabelsize, pad = 15)
plt.tick_params(axis='both', which = 'major', length=7, width=1.7)
plt.tick_params(axis='both', which = 'minor', length=4, width=1)
plt.xlabel('$\mathit{{\phi}_{g}}$', fontsize = axislabelsize, labelpad = 16)
plt.ylabel('$\mathregular{{V}_{0}/{V}_{g}}$ (mL)', fontsize = axislabelsize, labelpad = 12)
plt.legend(ncol=1, prop={'family': 'monospace', 'size': ticklabelsize}, handletextpad = 0.8, labelspacing=1)

# %% Save Viscosity and Displacement Data
saveData = False
if saveData == True:
    resultsFile = 'ViscData.csv'
    home = '/Users/alizz/Google Drive (alireza.meghdadi@gmail.com)/Southampton - GoogleDrive/Experiments/Blood Displacement/'
    with open((home + resultsFile),'w',newline='') as csvfile:
        fields =    ['technique','l:g ratio','flowrate','blood displacement','blood displacement std',
                      'bd per v gas','bd per v gass std','displacement ratio','displacement ratio std',
                      'viscosity','viscosity std']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for tech in displacement.keys():
            for lg in displacement[tech].keys():
                i = 0
                for flowrate in flowrates:
                    writer.writerow({
                        'technique'                     :       tech                                    ,
                        'l:g ratio'                     :       lg                                      ,
                        'flowrate'                      :       flowrate                                ,
                        'blood displacement'            :       displacement[tech][lg]['y'][i]          ,
                        'blood displacement std'        :       displacement[tech][lg]['err'][i]        ,
                        'bd per v gas'                  :       [x*(1/(float(lg[-1])+1)) for x in displacement[tech][lg]['y']][i] ,
                        'bd per v gass std'             :       [x*(1/(float(lg[-1])+1)) for x in displacement[tech][lg]['err']][i],
                        'displacement ratio'            :       displacementRatio[tech][lg]['y'][i]     ,
                        'displacement ratio std'        :       displacementRatio[tech][lg]['err'][i]   ,
                        'viscosity'                     :       mu[tech][lg][flowrate]['viscosity']     ,
                        'viscosity std'                 :       mu[tech][lg][flowrate]['viscosityError']})
                    # print(displacement[tech][lg]['x'][i]  == int(flowrate))             # Check that flowrate indices are correct
                    i += 1
    resultsFile = 'CMC Displaced.csv'
    home = '/Users/alizz/Google Drive (alireza.meghdadi@gmail.com)/Southampton - GoogleDrive/Experiments/Blood Displacement/'
    with open((home + resultsFile),'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Asymptotic Regression Fitting'])
        writer.writeheader()
        fields =   ['l:g','Y0','Y0_std','K','K_std']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        i = 0
        for lg in parameters.keys():
            writer.writerow({
                'l:g'               :       lg                                  ,
                'Y0'                :       plateauResults['Y0']['y'][i]        ,
                'Y0_std'            :       plateauResults['Y0']['std'][i]      ,
                'K'                 :       plateauResults['K']['y'][i]         ,
                'K_std'             :       plateauResults['K']['std'][i]       })
            i += 1
        writer.writerow({
                'l:g':'' ,'Y0':'','Y0_std':'','K':'','K_std' :''})
        writer = csv.DictWriter(csvfile, fieldnames=['V0 Linear Regression'])
        writer.writeheader()
        fields = ['slope','slope-std','intercept', 'intercept-std','p-slope','r2-pearson', 'p-pearson']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
                    'slope'         :       results['Y0'][0][0]                     ,
                    'slope-std'     :       np.sqrt(np.diag(results['Y0'][1]))[0]   ,
                    'intercept'     :       results['Y0'][0][1]                     ,
                    'intercept-std' :       np.sqrt(np.diag(results['Y0'][1]))[1]   ,
                    'p-slope'       :       plateauResults['Y0']['slope-p']         ,
                    'r2-pearson'    :       plateauResults['Y0']['r2']              ,
                    'p-pearson'     :       plateauResults['Y0']['pearson-p']       })
        writer.writerow({'slope':'','slope-std':'','intercept':'', 'intercept-std':'',
                         'p-slope':'','r2-pearson':'','p-pearson':''})
        writer = csv.DictWriter(csvfile, fieldnames=['K Linear Regression'])
        writer.writeheader()
        fields = ['slope','slope-std','intercept', 'intercept-std','p-slope','r2-pearson', 'p-pearson']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
                    'slope'         :       results['K'][0][0]                     ,
                    'slope-std'     :       np.sqrt(np.diag(results['K'][1]))[0]   ,
                    'intercept'     :       results['K'][0][1]                     ,
                    'intercept-std' :       np.sqrt(np.diag(results['K'][1]))[1]   ,
                    'p-slope'       :       plateauResults['K']['slope-p']         ,
                    'r2-pearson'    :       plateauResults['K']['r2']              ,
                    'p-pearson'     :       plateauResults['K']['pearson-p']       })

    resultsFile = 'CMC Displaced Per Volume Gas.csv'
    home = '/Users/alizz/Google Drive (alireza.meghdadi@gmail.com)/Southampton - GoogleDrive/Experiments/Blood Displacement/'
    with open((home + resultsFile),'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Asymptotic Regression Fitting'])
        writer.writeheader()
        fields =   ['l:g','Y0','Y0_std','K','K_std']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        i = 0
        for lg in parameters.keys():
            writer.writerow({
                'l:g'               :       lg                                  ,
                'Y0'                :       plateauResults2['Y0']['y'][i]        ,
                'Y0_std'            :       plateauResults2['Y0']['std'][i]      ,
                'K'                 :       plateauResults2['K']['y'][i]         ,
                'K_std'             :       plateauResults2['K']['std'][i]       })
            i += 1
        writer.writerow({
                'l:g':'' ,'Y0':'','Y0_std':'','K':'','K_std' :''})
        writer = csv.DictWriter(csvfile, fieldnames=['V0 Linear Regression'])
        writer.writeheader()
        fields = ['slope','slope-std','intercept', 'intercept-std','p-slope','r2-pearson', 'p-pearson']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
                    'slope'         :       results2['Y0'][0][0]                     ,
                    'slope-std'     :       np.sqrt(np.diag(results2['Y0'][1]))[0]   ,
                    'intercept'     :       results2['Y0'][0][1]                     ,
                    'intercept-std' :       np.sqrt(np.diag(results2['Y0'][1]))[1]   ,
                    'p-slope'       :       plateauResults2['Y0']['slope-p']         ,
                    'r2-pearson'    :       plateauResults2['Y0']['r2']              ,
                    'p-pearson'     :       plateauResults2['Y0']['pearson-p']       })
        writer.writerow({'slope':'','slope-std':'','intercept':'', 'intercept-std':'',
                         'p-slope':'','r2-pearson':'','p-pearson':''})
        writer = csv.DictWriter(csvfile, fieldnames=['K Linear Regression'])
        writer.writeheader()
        fields = ['slope','slope-std','intercept', 'intercept-std','p-slope','r2-pearson', 'p-pearson']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
                    'slope'         :       results2['K'][0][0]                     ,
                    'slope-std'     :       np.sqrt(np.diag(results2['K'][1]))[0]   ,
                    'intercept'     :       results2['K'][0][1]                     ,
                    'intercept-std' :       np.sqrt(np.diag(results2['K'][1]))[1]   ,
                    'p-slope'       :       plateauResults2['K']['slope-p']         ,
                    'r2-pearson'    :       plateauResults2['K']['r2']              ,
                    'p-pearson'     :       plateauResults2['K']['pearson-p']       })
# %% Save Figures
saveFigures = False
if saveFigures == True:
    titles = []
    for fig in figs:
        titles.append(fig.texts[0].get_text())
    home = '/Users/alizz/Google Drive (alireza.meghdadi@gmail.com)/Southampton - GoogleDrive/Experiments/Blood Displacement/Figures/'
    for num in range(0,len(figs)):
        if figs[num] == 0:
            continue
        figs[num].savefig(home+titles[num]+'.png',dpi=450,transparent=False)

