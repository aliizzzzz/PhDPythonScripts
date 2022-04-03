# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:33:39 2020

@author: alizz
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# %% Input
# Directory:
class cDir:
    dataDir     = '/Users/alizz/Google Drive (alireza.meghdadi@gmail.com)/Southampton - GoogleDrive/Experiments/Pipe Viscometry/Data/'
    resDir      = 'Preprocessed Results/'
    plawDir     = 'Processed Results/'
    prep        = dataDir + resDir
    pl          = dataDir + plawDir

SaveFigures     = False
Usety           = False
CorrectDp       = False
CorrectSlip     = False
quality         = 450
windowsize      = [24,12]

if CorrectSlip == True and CorrectDp == True:
    addTitle = 'Slip + DP Corrected '
elif CorrectSlip == True and CorrectDp == False:
    addTitle = 'Slip Corrected (DP Uncorrected) '
elif CorrectSlip == False and CorrectDp == True:
    addTitle = 'DP Corrected (Slip Uncorrected) '
else:
    addTitle = '(No Corrections) '

pipes       = ['2.48 mm PTFE','4.48 mm PTFE']
techniques  = ['DSS','TSS']
lgRatios    = ['1-3','1-4','1-5']

diameters = {'4.48 mm PTFE':4.48e-3,'2.48 mm PTFE':2.48e-3}
lengths = {'4.48 mm PTFE':15e-2,'2.48 mm PTFE':34e-2}

# %% Functions

def styleGraphs(graph, heading, xlimit, ylimit, xlab, ylab):
    graph.xaxis.set_major_locator(plt.MultipleLocator(50))
    graph.minorticks_on()
    graph.grid(True, linestyle=':',linewidth=1.5,which='minor')
    graph.grid(True, linestyle='-',linewidth=1.7,which='major')
    graph.set_ylim(ylimit)
    graph.set_xlim(xlimit)
    graph.tick_params(axis='both', labelsize=18)
    graph.tick_params(axis='both', which = 'major', length=7, width=1.7)
    graph.tick_params(axis='both', which = 'minor', length=4, width=1)
    graph.set_xlabel(xlab, fontsize = 20, labelpad = 20)
    graph.set_ylabel(ylab, fontsize = 20, labelpad = 15)
    graph.set_title(heading, pad=15, fontsize = 18, position=titlepos, weight = 'bold')
    graph.legend(prop={'family': 'monospace', 'size': 18}, handletextpad = 0)

def pLaw(shearRate, k, n, ty):
    tw = ty+(k*(shearRate**n))
    return tw

def pLawInv(tw,k,n, ty):
    shearRate = ((tw-ty)/k)**(1/n)
    return shearRate

def polyFit(tw,p):
    shearRate = (p[0]*tw**2)+(p[1]*tw)+p[2]
    return shearRate

def apparentVisc(shearRate, k, n, ty):
    vis = (ty/shearRate)+(k*(shearRate**(n-1)))
    return vis

def flowToShear (x,d):
    return ((x*32)/(60000000*np.pi*d**3))

def shearToFlow (x,d):
    return ((x*60000000*np.pi*d**3)/32)

def keyGen(p,t,r):
    return p + " | " + t + " | " + r

def calctw(dp,d,L):
    tw = [100*x*d/(4*L) for x in dp]
    return tw

# %% Read CSVs
n, k, nt, kt, ty, poly = {}, {}, {}, {}, {}, {}
nmin, nmax, kmin, kmax = {}, {}, {}, {}
if Usety == True:
    for file in (sorted(os.listdir(os.fsencode(cDir.pl)))):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename.startswith('Ty'):
            with open(cDir.pl + filename) as plfile:
                reader = csv.reader(plfile)
                nread, kread, tyread, p, t, r, pola, polb, polc = [], [], [], [], [], [], [], [], []
                i=0
                for row in reader:
                    if i == 0:
                        i += 1
                        continue
                    p.append(row[0])
                    t.append(row[1])
                    r.append(row[2])
                    nread.append(row[3])
                    kread.append(row[9])
                    tyread.append(row[15])
                    pola.append(row[17])
                    polb.append(row[18])
                    polc.append(row[19])

elif CorrectDp == True:
    for file in (sorted(os.listdir(os.fsencode(cDir.pl)))):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename.startswith('Corrected'):
            with open(cDir.pl + filename) as plfile:
                reader = csv.reader(plfile)
                nread, kread, tyread, p, t, r, pola, polb, polc = [], [], [], [], [], [], [], [], []
                nminread, nmaxread, kminread, kmaxread = [], [], [], []
                i=0
                for row in reader:
                    if i == 0:
                        i += 1
                        continue
                    p.append(row[0])
                    t.append(row[1])
                    r.append(row[2])
                    nread.append(row[3])
                    kread.append(row[9])
                    nminread.append(row[5])
                    nmaxread.append(row[7])
                    kminread.append(row[11])
                    kmaxread.append(row[13])
                    tyread.append(0)
                    pola.append(row[15])
                    polb.append(row[16])
                    polc.append(row[17])

    correctedCsvs = []
    bagelyCorrections = {}

    directory = os.fsencode(cDir.dataDir+'End-Effects/')
    directory_files = sorted(os.listdir(directory))
    for file in directory_files:
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):          # Open CSV file
            correctedCsvs.append(filename)
            with open(cDir.dataDir+'End-Effects/'+filename) as csvFile:
                reader = csv.reader(csvFile)
                yint = []
                for row in reader:
                    yint.append(row[1])
            bagelyCorrections[filename[:-4]]  = list(map(float,yint[1:]))

else:
    for file in (sorted(os.listdir(os.fsencode(cDir.pl)))):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename.startswith('Power'):
            with open(cDir.pl + filename) as plfile:
                reader = csv.reader(plfile)
                nread, kread, tyread, p, t, r, pola, polb, polc = [], [], [], [], [], [], [], [], []
                nminread, nmaxread, kminread, kmaxread = [], [], [], []
                i=0
                for row in reader:
                    if i == 0:
                        i += 1
                        continue
                    p.append(row[0])
                    t.append(row[1])
                    r.append(row[2])
                    nread.append(row[3])
                    kread.append(row[9])
                    nminread.append(row[5])
                    nmaxread.append(row[7])
                    kminread.append(row[11])
                    kmaxread.append(row[13])
                    tyread.append(0)
                    pola.append(row[15])
                    polb.append(row[16])
                    polc.append(row[17])


csvs        = []
directory = os.fsencode(cDir.prep)
directory_files = sorted(os.listdir(directory))
shearRate, obsShear ,appVisc, tw, correctedTw, correctedDp, twerr, dps  = {}, {}, {}, {}, {}, {}, {}, {}


for file in directory_files:
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):          # Open CSV file
        csvs.append(filename)
        with open(cDir.prep + filename) as csvFile:
            reader = csv.reader(csvFile)
            fMl, fM3, dp, SR, AV, SS, SSer = [],[],[],[],[],[],[]
            for row in reader:
                fMl.append(row[0])
                fM3.append(row[1])
                dp.append(row[2])
                SR.append(row[10])
                SS.append(row[11])
                SSer.append(row[14])

        flowRateMl                      = list(map(float,fMl[1:]))
        flowRateM3                      = list(map(float,fM3[1:]))
        dps[filename[:-4]]              = list(map(float,dp[1:]))
        shearRate[filename[:-4]]        = list(map(float,SR[1:]))
        tw[filename[:-4]]               = list(map(float,SS[1:]))
        twerr[filename[:-4]]            = list(map(float,SSer[1:]))

        pipe        = filename[:12]
        diameter    = float(filename[:4])/1000
        technique   = filename[15:18]
        lgRatio     = filename[21:24]

        key = keyGen(pipe,technique,lgRatio)
        if 'bagelyCorrections' not in locals():
            bagelyCorrections = {}
            for tech in techniques:
                for lg in lgRatios:
                    bagelyCorrections[tech + ' | '+ lg] = [0,0,0,0,0,0,0,0,0]
        correctedDp[key]    = np.subtract(dps[key],bagelyCorrections[technique + ' | '+ lgRatio])
        correctedTw[key]     = calctw(correctedDp[key], diameters[pipe], lengths[pipe])

obsShear[pipes[0]] = shearRate[keyGen(pipes[0],techniques[0],lgRatios[0])]
obsShear[pipes[1]] = shearRate[keyGen(pipes[1],techniques[0],lgRatios[0])]

for i in range (len(nread)):
    key = keyGen(p[i],t[i],r[i])
    n[key] = float(nread[i])
    k[key] = float(kread[i])
    ty[key]= float(tyread[i])
    poly[key] = [float(pola[i]),float(polb[i]),float(polc[i])]
    nmin[key] = float(nminread[i])
    nmax[key] = float(nmaxread[i])
    kmin[key] = float(kminread[i])
    kmax[key] = float(kmaxread[i])

# %% Plotting Parameters
maxQRange       = np.round(max(flowRateMl),-1)
maxShearRange2  = np.round(max(obsShear['2.48 mm PTFE'])+30,-1)
xShearlimit2    = [0,maxShearRange2]
maxShearRange4  = np.round(max(obsShear['4.48 mm PTFE'])+10,-1)
xShearlimit4    = [0,maxShearRange4]
maxtw           = np.round(max(max(correctedTw.values()))+10,-1)
xQlimit         = [0,maxQRange]
ytwlimit        = [0,maxtw]
yViscPoislimit  = {'2.48 mm PTFE':[0,0.4],'4.48 mm PTFE':[0,4]}
xdata1          = np.linspace(list(obsShear.values())[0][0],list(obsShear.values())[0][-1],1000)
xdata2          = np.linspace(list(obsShear.values())[1][0],list(obsShear.values())[1][-1],1000)
# xdata1          = np.linspace(list(obsShear.values())[0][0],1000,1000)
# xdata2          = np.linspace(list(obsShear.values())[1][0],1000,1000)
# xdata1          = np.linspace(0,list(obsShear.values())[0][-1],1000)
# xdata2          = np.linspace(0,list(obsShear.values())[1][-1],1000)
titlepos        = [0.5,1.2]
dpower          = 2
plt.style.use('ggplot')
figs    = []
titles  = []
fignum  = -1


# %% Figure 1 - Oldroyd-Jastrzebski Construction
if CorrectSlip == True:
    fignum  += 1
    if CorrectDp == True:
        titles.append('Figure '+ str(fignum+1) + ' | Pressure-Corrected Oldroyd-Jastrzebski Plots')
    else:
        titles.append('Figure '+ str(fignum+1) + ' | Uncorrected Oldroyd-Jastrzebski Construction')
    figs.append(plt.figure(figsize=[24,12]))
    figs[fignum].subplots_adjust(wspace=0.30, hspace=0.6)
    figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=26)
    graphs      = []
    graphs.append(figs[fignum].add_subplot(321))
    graphs.append(figs[fignum].add_subplot(322))
    graphs.append(figs[fignum].add_subplot(323))
    graphs.append(figs[fignum].add_subplot(324))
    graphs.append(figs[fignum].add_subplot(325))
    graphs.append(figs[fignum].add_subplot(326))
    i = -1
    x = []
    for p in pipes:
        x.append((1/((float(p[:4])/1000)**dpower)))

    gamma2   = {}
    twconst2 = {}
    beta     = {}
    yint     = {}
    xOldroydRange = {}
    xOldroydLimit = {}


    for tech in techniques:
        for lg in lgRatios:
            for pipe in pipes:
                key = keyGen(pipe, tech, lg)
                twconst2[key] = np.linspace(min(correctedTw[key]),max(correctedTw[key]),9)
                # twconst2[key] = np.array([5,10,15,20])
                # twconst2[key] = np.linspace(4,100,9)
                # twconst2[key] = pLaw(np.array(obsShear[pipe]),k[key],n[key],ty[key])

                # gamma2[key] = pLawInv(twconst2[key], k[key], n[key], ty[key])
                gamma2[key] = polyFit(twconst2[key], poly[key])
                # gamma2[key] = np.log(polyFit(twconst2[key], poly[key]))
    xOldroydRange   = np.round(max(1/((float(pipes[0][:4])/1000)**dpower),1/((float(pipes[1][:4])/1000)**dpower))+100,-5)
    xOldroydLimit   = [0,xOldroydRange/1000]

    for lg in lgRatios:
        for tech in techniques:
            head = tech + ' Foam, L:G ratio = ' + lg.replace('-',':')
            i +=1
            g = graphs[i]
            y = {}
            ms = []
            bs = []
            for count in range(len(list(twconst2.values())[0])):
                gammaPair = []
                for pipe in pipes:
                    key = keyGen(pipe, tech, lg)
                    gammaPair.append(gamma2[key][count])
                y[str(twconst2[key][count])] = gammaPair
            for key in y.keys():
                g.scatter([xx/1000 for xx in x],y[key],label = '%.1f'%(float(key))+' Pa')
                m ,b = np.polyfit([xx/1000 for xx in x],y[key],1)
                g.plot([0,xOldroydRange/1000],[x*m+b for x in [0,xOldroydRange/1000]],'k-',linewidth=0.5)
                ms.append(m/(8*float(key)))
                bs.append(b)
            beta[tech + ' | ' + lg] = ms
            yint[tech + ' | ' + lg] = bs
            yrange = [np.round(g.get_ylim()[0],-2),np.round(g.get_ylim()[1],-2)]
            styleGraphs(g,head,xOldroydLimit,yrange,r'$\mathregular{1/d^{2}}$ (${\times}1000$ '+'$\mathregular{m^{-2}})$','$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)')
            g.grid(False, which='minor')
            g.xaxis.set_major_locator(plt.MultipleLocator(25))
            g.legend(prop={'family': 'monospace', 'size': 14}, handletextpad = 0, ncol = 3)
    if min([min(x) for x in list(yint.values())]) < 0:
        CorrectSlip = False
        addTitle = 'DP Corrected (Slip Uncorrected) '
        fignum  = 1
        figs.append(0)
        titles.append('')

# %% Figure 2 - Slip Coefficent Plots
    if CorrectSlip == True:
        fignum  += 1
        figs.append(plt.figure(figsize=windowsize))
        if CorrectDp == True:
            titles.append('Figure '+ str(fignum+1) + ' | DP Corrected Slip Coefficeints')
        else:
            titles.append('Figure '+ str(fignum+1) + ' | Uncorrected Slip Coefficeints')
        figs[fignum].subplots_adjust(wspace=0.30, hspace=0.6)
        figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=26)
        g = figs[fignum].add_subplot(111)
        xdata = list(twconst2.values())[0]
        i = -1
        betareg = {}
        for key in beta.keys():
            i += 1
            graph = g.plot(xdata,beta[key],marker = 'x', markersize = 8, mew=2,linestyle = 'None', label=key[:3] + ' Foam, L:G ratio = ' + key[6:9].replace('-',':'))
            betareg[key] =np.polyfit(xdata,beta[key],1)
            g.set_xlim([8,16])
            g.plot(g.get_xlim(),[(x*betareg[key][0])+betareg[key][1] for x in g.get_xlim()], ':k', linewidth=1, color = graph[-1].get_color())
        g.set_xlabel(r'$\mathregular{\tau}_{w}$ ($\mathregular{Pa.s}$)',fontsize = 18, labelpad = 20)
        g.set_ylabel(r'$\mathregular{\beta}$ ($\mathregular{m^2}.{Pa^{-1}.s^{-1}}$)',fontsize = 18, labelpad = 15)
        g.tick_params(axis='both', labelsize=14)
        g.ticklabel_format(axis='y', style='sci', useMathText=True, scilimits = (0,0))
        g.legend(fontsize=14, ncol = 3)

    # %% Calculating Nominal Shear Rate
        trueShear   = {}
        for pipe in pipes:
            for tech in techniques:
                for lg in lgRatios:
                    key = keyGen(pipe, tech, lg)
                    # OJ Method:
                    eightTw   = [8*x for x in correctedTw[key]]
                    aTw       = [betareg[tech + ' | ' + lg][0]*x for x in correctedTw[key]]
                    b         = betareg[tech + ' | ' + lg][1]
                    slipShear = eightTw*(aTw+b) / (diameters[pipe]**dpower)
                    trueShear[key] = obsShear[pipe] - slipShear

else:
    fignum  = 1
    figs    = [0,0]
    titles  = ['','']
# %% Figure 3 - Effect of L:G ratio on Viscosity of Different Foams
fignum  += 1
titles.append('Figure '+ str(fignum+1) + ' | ' + addTitle + 'Effect of L:G ratio on Apparent Viscosity of Different Foams')
figs.append(plt.figure(figsize=windowsize))
figs[fignum].subplots_adjust(wspace=0.15, hspace=0.75)
figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=26)

g1 = figs[fignum].add_subplot(121)
g2 = figs[fignum].add_subplot(122)

for tech in techniques:
    if tech == 'DSS':
        g = g1
    else:
        g = g2
    for pipe in pipes:
        if pipe == '2.48 mm PTFE':
            x = xdata1
            mark = 's'
        else:
            x = xdata2
            mark = 'o'
        for lg in lgRatios:
            key = keyGen(pipe, tech, lg)
            if CorrectSlip == True:
                shearSmooth = np.linspace(list(trueShear[key])[0],list(trueShear[key])[-1],1000)
                shearData   =  trueShear[key]
            else:
                shearSmooth = x
                shearData   = obsShear[pipe]
            plot = g.plot(shearData,correctedTw[key], marker = mark, label = 'd = ' + pipe[:7] + ' | ' + lg.replace('-',':') + ' Foam', linestyle='None',markersize=7)
            g.errorbar(shearData,correctedTw[key],yerr=[x/2 for x in twerr[key]],fmt='None',ecolor=plot[-1].get_color(),elinewidth=1.5,capsize=4)
            g.plot(shearSmooth,pLaw(shearSmooth,k[key],n[key],ty[key]), linewidth=1, color=plot[-1].get_color())
styleGraphs(g1,'DSS Foam',xShearlimit2, ytwlimit,'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)',r'$\mathregular{\tau}_w$ (Pa)')
styleGraphs(g2,'TSS Foam',xShearlimit2, ytwlimit,'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)',r'$\mathregular{\tau}_w$ (Pa)')
g1.title.set_position([0.5,1.01])
g2.title.set_position([0.5,1.01])
g1.legend(prop={'family': 'monospace', 'size': 18}, handletextpad = 0, ncol=2, loc="lower right")
g2.legend(prop={'family': 'monospace', 'size': 18}, handletextpad = 0, ncol=2, loc="lower right")


# %% Figure 4 - Rheogram Comparison of Different Foams
fignum  += 1
titles.append('Figure '+ str(fignum+1) + ' | ' + addTitle + 'Rheogram Comparison of Different Foams')
figs.append(plt.figure(figsize=windowsize))
figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=26)
g = figs[fignum].add_subplot(111)
colours = []
for tech in techniques:
    i = 0
    for pipe in pipes:
        if pipe == '2.48 mm PTFE':
            x = xdata1
            mark = 's'
        else:
            x = xdata2
            mark = 'o'
        for lg in lgRatios:
            key = keyGen(pipe, tech, lg)
            if CorrectSlip == True:
                shearSmooth = np.linspace(list(trueShear[key])[0],list(trueShear[key])[-1],1000)
                shearData   =  trueShear[key]
            else:
                shearSmooth = x
                shearData   = obsShear[pipe]
            if tech == 'DSS':
                g.scatter(shearData,correctedTw[key], marker = mark, label =  'd = ' + pipe[:7] + ' | ' + lg.replace('-',':') + ' ' + tech + ' Foam', s=60)
                graph = g.plot(shearSmooth,pLaw(shearSmooth,k[key],n[key],ty[key]), linewidth=1)
                colours.append(graph[-1].get_color())
            else:
                g.scatter(shearData,correctedTw[key], marker = mark, color = colours[i], label =  'd = ' + pipe[:7] + ' | ' + lg.replace('-',':') + ' ' + tech + ' Foam', facecolor='none', linewidth=1.5, s=60)
                g.plot(shearSmooth,pLaw(shearSmooth,k[key],n[key],ty[key]), color = colours[i], linewidth=1, linestyle='dashed')
                i += 1
styleGraphs(g,'',xShearlimit2, ytwlimit,'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)',r'$\mathregular{\tau}_w$ (Pa)')
g.title.set_position([0.5,1.01])
g.legend(prop={'family': 'monospace', 'size': 16}, handletextpad = 0, ncol=4 , loc="lower right")
# g.set_yscale('log')
# g.set_xscale('log')

# %% Figure 5 - Viscosity (Apparent) Comparison of Different Foams
fignum  += 1
titles.append('Figure '+ str(fignum+1) + ' | ' + addTitle + 'Viscosity (Apparent) Comparison of Different Foams')
figs.append(plt.figure(figsize=windowsize))
figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=26)
figs[fignum].subplots_adjust(wspace=0.2, hspace=0.75)
g1 = figs[fignum].add_subplot(121)
g2 = figs[fignum].add_subplot(122)

for lg in lgRatios:
    for pipe in pipes:
        if pipe == '2.48 mm PTFE':
            x = xdata1
            g = g1
        else:
            x = xdata2
            g = g2
        for tech in techniques:
            key = keyGen(pipe, tech, lg)
            if CorrectSlip == True:
                shearSmooth = np.linspace(list(trueShear[key])[0],list(trueShear[key])[-1],1000)
            else:
                shearSmooth = x
            g.plot(shearSmooth,apparentVisc(shearSmooth,k[key],n[key],ty[key]), linewidth=1.5, label =  lg.replace('-',':') + ' ' + tech + ' Foam')
styleGraphs(g1,'2.48 mm PTFE',xShearlimit2, yViscPoislimit['2.48 mm PTFE'],'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)','$\mathregular{\mu}_{App}$ (Pa.s)')
styleGraphs(g2,'4.48 mm PTFE',xShearlimit4, yViscPoislimit['4.48 mm PTFE'],'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)','$\mathregular{\mu}_{App}$ (Pa.s)')
g1.title.set_position([0.5,1.01])
g2.title.set_position([0.5,1.01])
g2.xaxis.set_major_locator(plt.MultipleLocator(10))
g1.legend(prop={'family': 'monospace', 'size': 18}, handletextpad = 0.2, ncol=3)
g2.legend(prop={'family': 'monospace', 'size': 18}, handletextpad = 0.2, ncol=3)

# %% Figure 6 - Viscosity (Apparent) Master Graph

fignum  += 1
titles.append('Figure '+ str(fignum+1) + ' | ' + addTitle + 'Viscosity (Apparent) Master Graph')
figs.append(plt.figure(figsize=windowsize))
figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=26)
g = figs[fignum].add_subplot(111)
colours =[]

for pipe in pipes:
    i = 0
    if pipe == '2.48 mm PTFE':
        x = xdata1
    else:
        x = xdata2
    for tech in techniques:
        for lg in lgRatios:
            key = keyGen(pipe, tech, lg)
            if CorrectSlip == True:
                shearSmooth = np.linspace(list(trueShear[key])[0],list(trueShear[key])[-1],1000)
            else:
                shearSmooth = x
            if pipe == '2.48 mm PTFE':
                graph = g.plot(shearSmooth,apparentVisc(shearSmooth,k[key],n[key],ty[key]), linewidth=1.5, label = 'd = ' + pipe[:7] + ' | ' + lg.replace('-',':') + ' ' + tech + ' Foam')
                colours.append(graph[-1].get_color())
            else:
                g.plot(shearSmooth,apparentVisc(shearSmooth,k[key],n[key],ty[key]), color = colours[i], linestyle='dashed', linewidth=1.5, label = 'd = ' + pipe[:7] + ' | ' + lg.replace('-',':') + ' ' + tech + ' Foam')
                i += 1
styleGraphs(g,'',xShearlimit2, yViscPoislimit['4.48 mm PTFE'],'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)','$\mathregular{\mu}_{App}$ (Pa.s)')
g.title.set_position([0.5,1.01])
g.legend(prop={'family': 'monospace', 'size': 16}, handletextpad = 0.2, ncol=4)

# %% Figure 7 - Dependency of Wall Slip on Tube Diameter
fignum  += 1
titles.append('Figure '+ str(fignum+1) + ' | ' + addTitle + 'Dependency of Wall Slip on Tube Diameter')
figs.append(plt.figure(figsize=windowsize))
figs[fignum].subplots_adjust(wspace=0.15, hspace=0.75)
figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=26)
colours     = {'2.48 mm PTFE': 'g', '4.48 mm PTFE': (1.0,0.5 ,0)}
mark        = {'2.48 mm PTFE': 's', '4.48 mm PTFE': 'o'}
graphs      = []
graphs.append(figs[fignum].add_subplot(321))
graphs.append(figs[fignum].add_subplot(322))
graphs.append(figs[fignum].add_subplot(323))
graphs.append(figs[fignum].add_subplot(324))
graphs.append(figs[fignum].add_subplot(325))
graphs.append(figs[fignum].add_subplot(326))
i = -1

for lg in lgRatios:
    for tech in techniques:
        head = tech + ' Foam, L:G ratio = ' + lg.replace('-',':')
        i +=1
        g = graphs[i]
        for pipe in pipes:
            if pipe == '2.48 mm PTFE':
                x = xdata1
            else:
                x = xdata2
            key  = keyGen(pipe,tech,lg)
            roots = [np.roots([poly[key][0],poly[key][1],poly[key][2]-xx]) for xx in x]
            twpoly = [item[0] for item in roots]
            g.scatter(obsShear[pipe],correctedTw[key],marker=mark[pipe], color=colours[pipe], label ='d = ' + pipe[:7], facecolors='None', linewidth=1.5)
            g.plot(x,pLaw(x,k[key],n[key],ty[key]),'k-', linewidth=0.5)
            # g.plot(x,twpoly,'r-')

        styleGraphs(g,head,xShearlimit2,ytwlimit,'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)',r'$\mathregular{\tau}_w$ (Pa)')

# %% Figure 8 - Dependence of Viscosity on Shear-Emphasized Drainage (Master)
fignum  += 1
titles.append('Figure '+ str(fignum+1) + ' |  %Change in Viscosity due to Shear-Induced Drainage')
figs.append(plt.figure(figsize=windowsize))
figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=30)
g = figs[fignum].add_subplot(111)
colours =[]

for pipe in pipes:
    i = 0
    if pipe == '2.48 mm PTFE':
        x = xdata1
    else:
        x = xdata2
    for tech in techniques:
        for lg in lgRatios:
            key = keyGen(pipe, tech, lg)
            viscDiff = 100*(apparentVisc(x,kmax[key],nmax[key],ty[key]) - apparentVisc(x,kmin[key],nmin[key],ty[key]))/(2*apparentVisc(x,k[key],n[key],ty[key]))
            if pipe == '2.48 mm PTFE':
                graph = g.plot(x,viscDiff, linewidth=1.5, label = 'd = ' + pipe[:7] + ' | ' + lg.replace('-',':') + ' ' + tech + ' Foam')
                colours.append(graph[-1].get_color())
            else:
                g.plot(x,viscDiff, color = colours[i], linestyle='dashed', linewidth=1.5, label = 'd = ' + pipe[:7] + ' | ' + lg.replace('-',':') + ' ' + tech + ' Foam')
                i += 1

styleGraphs(g,'',xShearlimit2, [0,12],'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)','')
g.title.set_position([0.5,1.01])
g.legend(prop={'family': 'monospace', 'size': 16}, handletextpad = 0.2, ncol=4)
g.set_ylabel('%', fontsize = 24, labelpad = 15, rotation=0)
# g.set_yscale('log')
# g.set_xscale('log')

# %% Figure 9 - Dependence of Viscosity on Shear-Emphasized Drainage
fignum  += 1
titles.append('Figure '+ str(fignum+1) + ' | Dependence of Viscosity on Shear-Emphasized Drainage')
figs.append(plt.figure(figsize=windowsize))
figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=30)
g1 = figs[fignum].add_subplot(121)
g2 = figs[fignum].add_subplot(122)

for lg in lgRatios:
    for pipe in pipes:
        if pipe == '2.48 mm PTFE':
            x = xdata1
            g = g1
        else:
            x = xdata2
            g = g2
        for tech in techniques:
            key = keyGen(pipe, tech, lg)
            viscDiff = abs(apparentVisc(x,kmax[key],nmax[key],ty[key]) - apparentVisc(x,kmin[key],nmin[key],ty[key]))
            g.plot(x,viscDiff*1000, linewidth=1.5, label =  lg.replace('-',':') + ' ' + tech + ' Foam')

styleGraphs(g1,'2.48 mm PTFE',xShearlimit2, [-10,50],'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)','$\mathregular{\mu}_{App}$ (mPa.s)')
styleGraphs(g2,'4.48 mm PTFE',xShearlimit4, [0,1000],'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)','$\mathregular{\mu}_{App}$ (mPa.s)')
g1.title.set_position([0.5,1.01])
g2.title.set_position([0.5,1.01])
g2.xaxis.set_major_locator(plt.MultipleLocator(10))
g1.legend(prop={'family': 'monospace', 'size': 18}, handletextpad = 0, ncol=3)
g2.legend(prop={'family': 'monospace', 'size': 18}, handletextpad = 0, ncol=3)

# %% Figure 10 - Reynolds Number

fignum  += 1
titles.append('Figure '+ str(fignum+1) + ' | ' + addTitle + 'Reynolds Number')
figs.append(plt.figure(figsize=windowsize))
figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=26)
g = figs[fignum].add_subplot(111)
colours =[]
Re = {}

for pipe in pipes:
    i = 0
    if pipe == '2.48 mm PTFE':
        x = xdata1
    else:
        x = xdata2
    for tech in techniques:
        for lg in lgRatios:
            key = keyGen(pipe, tech, lg)
            flowSmooth = shearToFlow(x,diameters[pipe])/(6e7)
            shearSmooth = x
            rho = (1/(float(lg[2])+1)*997)+(1-(1/(float(lg[2])+1)))*1.225
            Re[key] = rho*flowSmooth*diameters[pipe]/(apparentVisc(shearSmooth,k[key],n[key],ty[key])*(np.pi*(diameters[pipe]/2)**2))
            if pipe == '2.48 mm PTFE':
                graph = g.plot(flowSmooth*6e7,Re[key], linewidth=1.5, label = 'd = ' + pipe[:7] + ' | ' + lg.replace('-',':') + ' ' + tech + ' Foam')
                colours.append(graph[-1].get_color())
            else:
                g.plot(flowSmooth*6e7,Re[key], color = colours[i], linestyle='dashed', linewidth=1.5, label = 'd = ' + pipe[:7] + ' | ' + lg.replace('-',':') + ' ' + tech + ' Foam')
                i += 1
styleGraphs(g,'',xQlimit, [0,2.5],'$\mathregular{Q}$ ($\mathregular{mL.min^{-1}}$)','$\mathregular{Re}$')
g.xaxis.set_major_locator(plt.MultipleLocator(4))
g.title.set_position([0.5,1.01])
g.legend(prop={'family': 'monospace', 'size': 16}, handletextpad = 0.2, ncol=4,loc=2)

# %% Figure 12 - Rheograms for the blood displacement paper
fignum  += 1
titles.append('Figure '+ str(fignum+1) + ' | ' + addTitle + 'Rheograms for the blood displacement paper')
figs.append(plt.figure(figsize=windowsize))
figs[fignum].subplots_adjust(wspace=0.2, hspace=0.75)
figs[fignum].suptitle(titles[fignum],weight='bold',fontsize=26)

g1 = figs[fignum].add_subplot(121)
g2 = figs[fignum].add_subplot(122)
colours = []
for tech in techniques:
    for pipe in pipes:
        if pipe == '2.48 mm PTFE':
            continue
        else:
            x = xdata2
        i = 0
        for lg in lgRatios:

            key = keyGen(pipe, tech, lg)
            if CorrectSlip == True:
                shearSmooth = np.linspace(list(trueShear[key])[0],list(trueShear[key])[-1],1000)
                shearData   =  trueShear[key]
            else:
                shearSmooth = x
                shearData   = obsShear[pipe]
            if tech == 'DSS':
                g1.scatter(shearData,correctedTw[key], marker = 'o', label = lg.replace('-',':') + ' ' + tech + ' Foam', linestyle='None',s=60)
                plot = g1.plot(shearSmooth,pLaw(shearSmooth,k[key],n[key],ty[key]), linewidth=1)
                g2.plot(shearSmooth,apparentVisc(shearSmooth,k[key],n[key],ty[key]), linewidth=1.5, label =  lg.replace('-',':') + ' ' + tech + ' Foam')
                g1.errorbar(shearData,correctedTw[key],yerr=[x/2 for x in twerr[key]],fmt='None',elinewidth=1.5,capsize=4, color = plot[-1].get_color())
                colours.append(plot[-1].get_color())
            else:
                g1.scatter(shearData,correctedTw[key], marker = 's', label = lg.replace('-',':') + ' ' + tech + ' Foam', linestyle='None',s=60, facecolor =plt.gca().get_facecolor(), color = colours[i] ,zorder = 6, linewidth=1.5)
                plot = g1.plot(shearSmooth,pLaw(shearSmooth,k[key],n[key],ty[key]), linewidth=1, color = colours[i], linestyle = 'dashed')
                g2.plot(shearSmooth,apparentVisc(shearSmooth,k[key],n[key],ty[key]), linewidth=1.5, label =  lg.replace('-',':') + ' ' + tech + ' Foam', color = colours[i], linestyle = 'dashed')
                g1.errorbar(shearData,correctedTw[key],yerr=[x/2 for x in twerr[key]],fmt='None',elinewidth=1.5,capsize=4, color = colours[i])
            i += 1
styleGraphs(g1,'Rheogram',[0,80], ytwlimit,'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)',r'$\mathregular{\tau}_w$ (Pa)')
styleGraphs(g2,'Apparent Viscosity',xShearlimit4, yViscPoislimit['4.48 mm PTFE'],'$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)','$\mathregular{\mu}_{App}$ (Pa.s)')
g1.legend(prop={'family': 'monospace', 'size': 18}, handletextpad = 0, ncol=2, loc="lower right")
g2.legend(prop={'family': 'monospace', 'size': 18}, handletextpad = 0, ncol=2, loc="lower right")
g1.xaxis.set_major_locator(plt.MultipleLocator(10))
g2.xaxis.set_major_locator(plt.MultipleLocator(10))

# %% Save Figures
if SaveFigures == True:
    for num in range(0,fignum+1):
        if figs[num] == 0:
            continue
        figs[num].savefig((cDir.pl + '/' + titles[num] + '.png'),dpi=quality, transparent = False)

# %% Get Visc Ranges
shear={'2.48 mm PTFE':[xdata1[0],xdata1[-1]],'4.48 mm PTFE':[xdata2[0],xdata2[-1]]}
viscRanges = {}
twRanges = {}
for pipe in pipes:
    for tech in techniques:
        for lg in lgRatios:
            key = keyGen(pipe,tech,lg)
            viscRanges[key] = (apparentVisc(x,k[key],n[key],ty[key]) for x in shear[pipe])
            twRanges[key] = (correctedTw[key][0], correctedTw[key][-1])

np.max(list({k:v for (k,v) in twRanges.items() if 'DSS' in k and '4.48' in k}.values()))


