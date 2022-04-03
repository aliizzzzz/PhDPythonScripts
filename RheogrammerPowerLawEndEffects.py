 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 23:00:20 2019

@author: alizz
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy import stats

# %% Input
# Directory:
class cDir:
    dataDir     = '/Users/alizz/Google Drive (alireza.meghdadi@gmail.com)/Southampton - GoogleDrive/'
    dataDir     += 'Experiments/Pipe Viscometry/Data/'
    prepDir     = 'Preprocessed Results/'
    full        = dataDir + prepDir

SaveFigures     = False
SavePoLaw       = False
quality         = 450

pipes       = ['2.48 mm PTFE','4.48 mm PTFE']
techniques  = ['DSS','TSS']
lgRatios    = ['1-3','1-4','1-5']

diameters = {'4.48 mm PTFE':4.48e-3,'2.48 mm PTFE':2.48e-3}
lengths = {'4.48 mm PTFE':15e-2,'2.48 mm PTFE':34e-2}

# %% Functions
def plotCurves (graph,x,y, heading, xlimit, ylimit, xlab, ylab,datalab):
    graph.plot(x,y,marker = 's', markersize = 6, linestyle = 'None', color='k', label = datalab)
    if pipe == '2.48 mm PTFE':
        graph.xaxis.set_major_locator(plt.MultipleLocator(50))
    else:
        graph.xaxis.set_major_locator(plt.MultipleLocator(10))
    graph.minorticks_on()
    graph.grid(True, which='both')
    graph.grid(linestyle=':',linewidth=0.5,which='minor')
    # graph.xaxis.set_minor_locator(plt.MultipleLocator(5))
    graph.set_ylim(ylimit)
    graph.set_xlim(xlimit)
    graph.tick_params(axis='both', labelsize=14)
    graph.set_xlabel(xlab, fontsize = 18, labelpad = 20)
    graph.set_ylabel(ylab, fontsize = 18, labelpad = 15)
    graph.title.set_text(heading)
    graph.title.set_position(titlepos)
    graph.title.set_fontsize(18)

def pLaw(shearRate, k, n):
    tw = (k*(shearRate**n))
    return tw

def polyFit(x,p):
    y = (p[0]*x**2)+(p[1]*x)+p[2]
    return y

def apparentVisc(shearRate, k, n):
    vis = (k*(shearRate**(n-1)))
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

# %% Read End-Effect Corrections
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


# %% Read CSV
csvs        = []
params      = []
polyParams  = []
paramMins   = []
paramMaxs   = []
perrs       = []
perrMins    = []
perrMaxs    = []
chi         = []
ks          = []
pearson     = []
dps         = {}
figs        = []

directory = os.fsencode(cDir.full)
directory_files = sorted(os.listdir(directory))
for file in directory_files:
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):          # Open CSV file
        csvs.append(filename)
        with open(cDir.full + filename) as csvFile:
            reader = csv.reader(csvFile)

            flowRateMl, flowRateM3, viscPoiseuille,viscPoiseuilleStd = [],[],[],[]
            obsShear, tw, twStd = [],[],[]
            viscPoiseuilleMin, viscPoiseuilleMax = [], []
            viscPoiseuilleSemMin, viscPoiseuilleSemMax = [], []
            twMin, twMax, twSemMin, twSemMax, dpAveRep, dpAveStd = [], [], [], [], [], []

            for row in reader:
                flowRateMl.append(row[0])
                flowRateM3.append(row[1])
                dpAveRep.append(row[2])
                dpAveStd.append(row[3])
                viscPoiseuille.append(row[4])
                viscPoiseuilleMin.append(row[5])
                viscPoiseuilleMax.append(row[6])
                viscPoiseuilleStd.append(row[7])
                viscPoiseuilleSemMin.append(row[8])
                viscPoiseuilleSemMax.append(row[9])
                obsShear.append(row[10])
                tw.append(row[11])
                twMin.append(row[12])
                twMax.append(row[13])
                twStd.append(row[14])
                twSemMin.append(row[15])
                twSemMax.append(row[16])

        flowRateMl                      = list(map(float,flowRateMl[1:]))
        flowRateM3                      = list(map(float,flowRateM3[1:]))
        dpAveRep                        = list(map(float,dpAveRep[1:]))
        dpAveStd                        = list(map(float,dpAveStd[1:]))
        viscPoiseuille                  = list(map(float,viscPoiseuille[1:]))
        viscPoiseuilleMin               = list(map(float,viscPoiseuilleMin[1:]))
        viscPoiseuilleMax               = list(map(float,viscPoiseuilleMax[1:]))
        viscPoiseuilleStd               = list(map(float,viscPoiseuilleStd[1:]))
        viscPoiseuilleSemMin            = list(map(float,viscPoiseuilleSemMin[1:]))
        viscPoiseuilleSemMax            = list(map(float,viscPoiseuilleSemMax[1:]))
        obsShear                        = list(map(float,obsShear[1:]))
        tw                              = list(map(float,tw[1:]))
        twMin                           = list(map(float,twMin[1:]))
        twMax                           = list(map(float,twMax[1:]))
        twStd                           = list(map(float,twStd[1:]))
        twSemMin                        = list(map(float,twSemMin[1:]))
        twSemMax                        = list(map(float,twSemMax[1:]))

        pipe        = filename[:12]
        diameter    = float(filename[:4])/1000
        technique   = filename[15:18]
        lgRatio     = filename[21:24]

        key = keyGen(pipe,technique,lgRatio)
        dps[key] = dpAveRep
        correctedDp     = np.subtract(dpAveRep,bagelyCorrections[technique + ' | '+ lgRatio])
        correctedTw     = calctw(correctedDp, diameters[pipe], lengths[pipe])
        twOffset        = np.subtract(tw,correctedTw)
        correctedTwMin  = np.subtract(twMin,twOffset)
        correctedTwMax  = np.subtract(twMax,twOffset)

# %% Power-Law Fitting
        
        # twStd was halved bcause preprocessing-V5 stores 2*standard deviations in twStd
        param, cov = fit(pLaw,obsShear,correctedTw,bounds=(0,[np.inf,1]),
                         sigma=[(x/2) for x in twStd], absolute_sigma=True)
        paramMin, covMin = fit(pLaw,obsShear,correctedTwMin,bounds=(0,[np.inf,1]),
                               sigma=twSemMin,absolute_sigma=True)
        paramMax, covMax = fit(pLaw,obsShear,correctedTwMax,bounds=(0,[np.inf,1]),
                               sigma=twSemMax,absolute_sigma=True)

        perr = np.sqrt(np.diag(cov))
        up = param + 2*perr # upper limit power law paramers
        down = param - 2*perr # lower limit power law paramers

        perrMin = np.sqrt(np.diag(covMin))
        perrMax = np.sqrt(np.diag(covMax))

        params.append(param)
        paramMins.append(paramMin)
        paramMaxs.append(paramMax)
        perrs.append(perr)
        perrMins.append(perrMin)
        perrMaxs.append(perrMax)

# %% Polynomial Fitting
        polyParam = np.polyfit(correctedTw,obsShear,2)
        polyParams.append(polyParam)

# %% Rheograms

        maxQRange       = np.round(max(flowRateMl),-1)
        xextra          = {'2.48 mm PTFE':30,'4.48 mm PTFE':10}
        maxShearRange   = np.round(max(obsShear)+xextra[pipe],-1)
        maxtw           = {'2.48 mm PTFE':40,'4.48 mm PTFE':80}
        xQlimit         = [0,maxQRange]
        xShearlimit     = [0,maxShearRange]
        ytwlimit        = [0,maxtw[pipe]]
        yViscPoislimit  = {'2.48 mm PTFE':[0,0.4],'4.48 mm PTFE':[0,4]}
        yDplimit        = {'2.48 mm PTFE':[0,160],'4.48 mm PTFE':[0,100]}
        xdata           = np.linspace(obsShear[0],obsShear[-1],500)
        titlepos        = [0.5,1.1]

        twFitData   = pLaw(xdata, *param)
        twup        = pLaw(xdata, *up)
        twdown      = pLaw(xdata, *down)
        twFitMin    = pLaw(xdata, *paramMin)
        twFitMax    = pLaw(xdata, *paramMax)
        appVisc     = apparentVisc(xdata, *param)
        appViscMin  = apparentVisc(xdata, *paramMin)
        appViscMax  = apparentVisc(xdata, *paramMax)

        title       = pipe + ' - ' + technique + ' Foam; Liquid to Gas Ratio = ' + lgRatio[0]
        title       += ':' + lgRatio[-1]

        ymod = [x for x in pLaw(obsShear, *param)]
        var = [(x/2)**2 for x in twStd]
        # varNorm = [abs((x-max(var))/(max(var)-min(var))) for x in var]

        chi.append(stats.chisquare(correctedTw, f_exp=ymod,ddof = 7))
        ks.append(stats.ks_2samp(correctedTw,ymod))              # Kolmogorov - Smirnoff test
        pearson.append(stats.pearsonr(np.log(obsShear),np.log(correctedTw))) #Pearson Test

        figs.append(plt.figure(figsize=[24, 24]))
        figs[-1].subplots_adjust(wspace=0.25, hspace=0.5)
        figs[-1].suptitle(title,weight='bold',fontsize=30)
        plt.style.use('ggplot')

        # g1.grid(False, which='both')
        g1 = figs[-1].add_subplot(221)

        g2 = figs[-1].add_subplot(222)
        g2title = 'Pressure Difference Averages'
        plotCurves(g2, flowRateMl, dpAveRep, g2title, xQlimit, yDplimit[pipe],
                   'Q ($\mathregular{mL.min^{-1}}$)', '$\mathregular{\Delta}$P (mBar)',
                   'Original Data')
        g2.plot(flowRateMl,correctedDp, marker = 'o', markersize = 6, linestyle = 'None',
                color='g', label = 'Corrected Data')
        g2.errorbar(flowRateMl,dpAveRep,yerr=dpAveStd,fmt='None',ecolor='black',elinewidth=1,
                    capsize=4,zorder=6)
        g2.errorbar(flowRateMl,correctedDp,yerr=dpAveStd,fmt='None',ecolor='green',elinewidth=1,
                    capsize=4,zorder=6)
        g2.xaxis.set_major_locator(plt.MultipleLocator(4))
        g2.legend(fontsize=12)

        g3 = figs[-1].add_subplot(223)
        g3title = 'Rheogram'
        plotCurves (g3,obsShear,tw, g3title, xShearlimit, ytwlimit,
                    '$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)',
                    r'$\mathregular{\tau}_w$ (Pa)',r'$\mathregular{\tau}_{w_{ave}}$')
        g3.plot(obsShear,correctedTw,marker = 'o',markersize = 6, linestyle = 'None', color='g',
                label = r'Corrected $\mathregular{\tau}_{w_{ave}}$')
        g3.errorbar(obsShear,tw,yerr=twStd,fmt='None',ecolor='black',elinewidth=1,capsize=4)
        g3.errorbar(obsShear,correctedTw,yerr=twStd,fmt='None',ecolor='green',elinewidth=1,capsize=4)
        g3.plot(xdata, twFitData, 'k--',label='Power-Law Fit, ' +
                '$\mathregular{R^2}$ = %.3f' % pearson[-1][0])
        # ydata = np.linspace(correctedTw[0],correctedTw[-1],500)
        # g3.plot(polyFit(ydata,polyParam),ydata, '-r')
        g3.fill_between(xdata,twdown,twup, alpha=0.1, color='m',label='95% Confidence Interval')
        g3.legend(fontsize=12)

        g4 = figs[-1].add_subplot(224)
        g4title     = 'Power-Law Fit, ' + '$\mathregular{R^2}$ = %.3f' % pearson[-1][0]
        g4.grid(True, which='both')
        g4.grid(linestyle=':',linewidth=0.5,which='minor')
        yax2        = g4.twinx()
        yax2.grid(False, which='both')
        twFitLine   = g4.plot(xdata, twFitData, 'g--',label=g4title)
        twDot       = g4.plot(obsShear,correctedTw,marker = 's', markersize = 4, linestyle = 'None',
                              color='k', label = r'$\mathregular{\tau}_{w_{ave}}$')
        twMaxDot    = g4.plot(obsShear,correctedTwMax,marker = '^', markersize = 4,
                              linestyle = 'None', color='red',
                              label = r'$\mathregular{\tau}_{w_{max}}$')
        twMinDot    = g4.plot(obsShear,correctedTwMin,marker = 'v', markersize = 4,
                              linestyle = 'None', color='blue',
                              label = r'$\mathregular{\tau}_{w_{min}}$')
        viscLine    = yax2.plot(xdata, appVisc,'r--', label='Apparent Viscosity')
        region      = g4.fill_between(xdata,twFitMin,twFitMax, alpha=0.2, color='g',
                                      label='Shear Stress Min-Max Region')
        regionVisc  = yax2.fill_between(xdata,appViscMax,appViscMin, alpha=0.2, color='r',
                                        label='Viscosity Min-Max Region')
        yax2.set_ylabel('$\mathregular{\mu}_{app}$ (Pa.s)',fontsize = 18, labelpad = 15)
        yax2.tick_params(axis='both', labelsize=14)
        yax2.set_ylim(yViscPoislimit[pipe])
        xax2Ticks   = np.round([shearToFlow(x,diameter) for x in g4.axes.get_xticks()],1)
        xax2        = g4.twiny()
        xax2.set_xticks(xax2Ticks)
        xax2.axes.set_xlim([shearToFlow(x,diameter) for x in xShearlimit])
        xax2.set_xlabel('Q ($\mathregular{mL.min^{-1}}$)', fontsize = 18, labelpad = 20)
        xax2.tick_params(axis='both', labelsize=14)
        xax2.grid(False, which='both')

        if pipe == '2.48 mm PTFE':
            g4.xaxis.set_major_locator(plt.MultipleLocator(50))
        else:
            g4.xaxis.set_major_locator(plt.MultipleLocator(10))
        g4.minorticks_on()
        g4.set_ylim([0,20])
        g4.set_xlim(xShearlimit)
        g4.tick_params(axis='both', labelsize=14)
        g4.set_xlabel('$\mathcal{\.\gamma}_{obs}$ ($\mathregular{s^{-1}}$)', fontsize = 18,
                      labelpad = 20)
        g4.set_ylabel(r'$\mathregular{\tau}_w$ (Pa)', fontsize = 18, labelpad = 15)
        lns = twFitLine + viscLine
        lns.append(region)
        lns.append(regionVisc)
        lns += twDot + twMinDot + twMaxDot
        labs = [l.get_label() for l in lns]
        g4.legend(lns, labs, fontsize=12, ncol=2)

        if SaveFigures == True:
            folder = 'Rheograms/'
            if os.path.exists(cDir.dataDir+folder) == False:
                    os.mkdir(cDir.dataDir+folder)

            nameEnd = 'Pressure-Corrected Power Law Curve - ' + filename[:-4]
            figs[-1].savefig((cDir.dataDir+folder + '/' + nameEnd + '.png'),dpi=quality,
                             transparent = False)

# %% Save PowerLaw Indices
if SavePoLaw == True:
    # Corrected data:

    # Uncorrected data
    resultsFile = 'Corrected Power Law Indices.csv'
    with open((cDir.dataDir + 'Processed Results/' + resultsFile),'w',newline='') as csvfile:
        fields = ['pipe'                        ,
                  'technique'                   ,
                  'l:g ratio'                   ,
                  'n (flow index)'              ,
                  'n error'                     ,
                  'min n (flow index)'          ,
                  'min n error'                 ,
                  'max n (flow index)'          ,
                  'max n error'                 ,
                  'k (consistency index)'       ,
                  'k error'                     ,
                  'min k (consistency index)'   ,
                  'min k error'                 ,
                  'max k (consistency index)'   ,
                  'max k error'                 ,
                  'polynomial a'                ,
                  'polynomial b'                ,
                  'polynomial c'                ,
                  'chisquared'                  ,
                  'chi-p-value'                 ,
                  'KS D stats'                  ,
                  'KS p-value'                  ,
                  'pearson-p'                   ,
                  'r2'                          ]

        writer = csv.DictWriter(csvfile, fieldnames = fields)
        writer.writeheader()
        for i in range(len(csvs)):
            writer.writerow({
                      'pipe'                        :   csvs[i][:12]        ,
                      'technique'                   :   csvs[i][15:18]      ,
                      'l:g ratio'                   :   csvs[i][21:24]      ,
                      'n (flow index)'              :   params[i][1]        ,
                      'n error'                     :   perrs[i][1]         ,
                      'min n (flow index)'          :   paramMins[i][1]     ,
                      'min n error'                 :   perrMins[i][1]      ,
                      'max n (flow index)'          :   paramMaxs[i][1]     ,
                      'max n error'                 :   perrMaxs[i][1]      ,
                      'k (consistency index)'       :   params[i][0]        ,
                      'k error'                     :   perrs[i][0]         ,
                      'min k (consistency index)'   :   paramMins[i][0]     ,
                      'min k error'                 :   perrMins[i][0]      ,
                      'max k (consistency index)'   :   paramMaxs[i][0]     ,
                      'max k error'                 :   perrMaxs[i][0]      ,
                      'polynomial a'                :   polyParams[i][0]    ,
                      'polynomial b'                :   polyParams[i][1]    ,
                      'polynomial c'                :   polyParams[i][2]    ,
                      'chisquared'                  :   chi[i][0]           ,
                      'chi-p-value'                 :   chi[i][1]           ,
                      'KS D stats'                  :   ks[i][0]            ,
                      'KS p-value'                  :   ks[i][1]            ,
                      'pearson-p'                   :   pearson[i][1]       ,
                      'r2'                          :   pearson[i][0]       })


