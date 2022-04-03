#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:45:19 2021

@author: alizz
"""
import os
import numpy as np
from IPython import get_ipython


# %% Functions
def cross2(shearRate, mu0, lamb, n):
    mueff = mu0/(1+((lamb*shearRate)**(1-n)))
    return mueff


def flowToShear(x, d):
    return ((x*32)/(60000000*np.pi*d**3))


# %% Input Parameters
defaultRhoRatio = 1.45677361557583
rhoRatio        = 1
rhoRange        = 0.1
conditions = {'compressibility'     :    True      ,
              'outflow'             :    True       ,
              'slurm'               :    True      }
user = "/am6e18"

cmc = {"mu0"    :       0.019       ,
       "n"      :       0.67        ,
       "lamb"   :       0.01
       }

dp = {"02": 102587.426104,
      "05": 102949.910894,
      "10": 103496.775784}
foams = {"DSS 1-3": {"K": 11.9293291220738, "n": 0.2819, "rho": 250.16875},
         "DSS 1-4": {"K": 8.14170811387433, "n": 0.4453, "rho": 200.38000},
         "DSS 1-5": {"K": 11.2683735402322, "n": 0.4062, "rho": 167.18750},
         "TSS 1-4": {"K": 10.2676687287686, "n": 0.3929, "rho": 200.38000},
         "TSS 1-5": {"K": 9.78646197990849, "n": 0.4237, "rho": 167.18750}}

compressibility = {"02": {"DSS 1-3": {"rho0": 250.16875, "rho": 300.742990751004,
                                      "rhoRatio": 1.202160504663, "p0": 101325,
                                      "p": 204910.298423058, "K0": 420190.670809563,
                                      "K": 734028.8163805, "n": 3.029755673},
                          "DSS 1-4": {"rho0": 200.38, "rho": 247.072918355005,
                                      "rhoRatio": 1.233021850260, "p0": 101325,
                                      "p": 204910.298423058, "K0": 371779.235309664,
                                      "K": 641673.0294355, "n": 2.605547349},
                          "DSS 1-5": {"rho0": 167.1875, "rho": 206.202414823514,
                                      "rhoRatio": 1.233360238197, "p0": 101325,
                                      "p": 204910.298423058, "K0": 370333.005969698,
                                      "K": 642206.0083410, "n": 2.624679910},
                          "TSS 1-4": {"rho0": 200.38, "rho": 246.484204443758,
                                      "rhoRatio":1.230083862879, "p0": 101325,
                                      "p": 204910.298423058, "K0" : 375535.697975269,
                                      "K": 649800.5686524, "n": 2.647796726},
                          "TSS 1-5": {"rho0" : 167.1875, "rho": 207.908787631748,
                                      "rhoRatio" : 1.243566580227, "p0" : 101325,
                                      "p" : 204910.298423058, "K0" : 356338.049327148,
                                      "K" : 617895.9391615, "n" : 2.525145888}},

                   "05": {"DSS 1-3": {"rho0" :250.16875, "rho" :300.903182535393, "rhoRatio" :1.202800839575, "p0" : 101325, "p" : 205301.492782427, "K0" : 420190.670809563, "K" : 735214.0397102, "n" : 3.029755673}   ,
                          "DSS 1-4": {"rho0" :200.38   , "rho" :247.223472087070, "rhoRatio" :1.233773191372, "p0" : 101325, "p" : 205301.492782427, "K0" : 371779.235309664, "K" : 642692.3048614, "n" : 2.605547349}   ,
                          "DSS 1-5": {"rho0" :167.1875 , "rho" :206.327959180545, "rhoRatio" :1.234111157715, "p0" : 101325, "p" : 205301.492782427, "K0" : 370333.005969698, "K" : 643232.7683169, "n" : 2.624679910}   ,
                          "TSS 1-4": {"rho0" :200.38   , "rho" :246.632519865246, "rhoRatio" :1.230824033662, "p0" : 101325, "p" : 205301.492782427, "K0" : 375535.697975269, "K" : 650836.3717966, "n" : 2.647796726}   ,
                          "TSS 1-5": {"rho0" :167.1875 , "rho" :208.040352684521, "rhoRatio" :1.244353511384, "p0" : 101325, "p" : 205301.492782427, "K0" : 356338.049327148, "K" : 618883.7619896, "n" : 2.525145888}   }, 
                  
                   "10": {"DSS 1-3": {"rho0" :250.16875, "rho" :301.147837920790, "rhoRatio" :1.203778800992, "p0" : 101325, "p" : 205899.766753022, "K0" : 420190.670809563, "K" : 737026.6636668, "n" : 3.029755673}   ,
                          "DSS 1-4": {"rho0" :200.38   , "rho" :247.453437470830, "rhoRatio" :1.234920837762, "p0" : 101325, "p" : 205899.766753022, "K0" : 371779.235309664, "K" : 644251.1360193, "n" : 2.605547349}   ,
                          "DSS 1-5": {"rho0" :167.1875 , "rho" :206.519721026566, "rhoRatio" :1.235258144458, "p0" : 101325, "p" : 205899.766753022, "K0" : 370333.005969698, "K" : 644803.0459881, "n" : 2.624679910}   ,
                          "TSS 1-4": {"rho0" :200.38   , "rho" :246.859062540079, "rhoRatio" :1.231954598962, "p0" : 101325, "p" : 205899.766753022, "K0" : 375535.697975269, "K" : 652420.4796574, "n" : 2.647796726}   ,
                          "TSS 1-5": {"rho0" :167.1875 , "rho" :208.241316907658, "rhoRatio" :1.245555540382, "p0" : 101325, "p" : 205899.766753022, "K0" : 356338.049327148, "K" : 620394.4910466, "n" : 2.525145888}   }}


if conditions['slurm'] == True:
    ppn             = 40
    nodes           = 15
    fluent          ="21.2"
else:
    ppn             = 16
    nodes           = 32
    fluent          = "20.1"
walltime        = {"02" : "15:00:00"    ,
                   "05" : "30:00:00"    ,
                   "10" : "60:00:00"    }
courant         = 0.75
minSize         = 0.01e-3 #m
totalData       = 50
maxIterations   = 60
flowTime        = 5
techniques      = ["DSS","TSS"]
lgs             = ["1-3","1-4","1-5"]
flowrates       = [2,5,10]
submission            = ""

for tech in techniques:
    for lg in lgs:
        for q in [str(x).zfill(2) for x in flowrates]:
            if tech == "TSS" and lg == "1-3":
                continue

            needleDiam      = (0.514*1e-3 )/2
            linearVel       = float(q)/(60000000*np.pi*(needleDiam)**2) #m/s
            stepSize        = round(courant * minSize /linearVel,8)
            nSteps          = int(round(flowTime/stepSize,-1))
            saveFrequency   = int(nSteps/totalData)
            shear           = flowToShear(int(q), 4.48/1000)
            mu              = cross2(shear, cmc["mu0"], cmc["lamb"], cmc["n"])

            # %% IRIDIS Paths
            if conditions['slurm'] == True:
                iridisHome      = "/mainfs/scratch" +user
            else:
                iridisHome      = "/scratch" +user
            experimentDir   = "/Ansys/SimpleTube/"
            targetDir       = iridisHome + experimentDir
            fileID          = tech+lg+"-"+q+"mlpm--"+"%.2E"%(stepSize)+"--"+str(flowTime)+"s"
            caseName        = "SimpleTube.cas.gz"
            journalName     = "journal_" +tech+lg+"-"+q+"mlpm.jou"
            runName         = "run_fluent_" +tech+lg+"-"+q+"mlpm"
            caseLoc         = tech + "-" + lg +"/"+q+"mlpm/"
            dataLoc         = "dataFiles/"
            saveLoc         = "reportFiles/"

            # %% .jou Compiler
            if conditions['compressibility'] == True:
                setmaterial     = "; Set foam properties\n/define/materials change-create foam foam yes compressible-liquid " + str(compressibility[q][tech + " " + lg]["p0"]) + " " + str(compressibility[q][tech + " " + lg]["rho0"]) + " " + str(compressibility[q][tech + " " + lg]["K0"]) + " " + str(compressibility[q][tech + " " + lg]["n"]) + " " + str(compressibility[q][tech + " " + lg]["rhoRatio"]+ rhoRange) + " "  + str(compressibility[q][tech + " " + lg]["rhoRatio"]- rhoRange) + " no no yes herschel-bulkley shear-rate-dependent " + str(foams[tech + " " + lg]["K"]) + " " + str(foams[tech + " " + lg]["n"]) + " 0 1 no no no"
            else:
                setmaterial     = "; Set foam properties\n/define/materials change-create foam foam yes constant " + str(foams[tech + " " + lg]["rho"]) + " no no yes herschel-bulkley shear-rate-dependent " + str(foams[tech + " " + lg]["K"]) + " " + str(foams[tech + " " + lg]["n"]) + " 0 1 no no no"
            setboundary     = "; Set boundary conditions\n/define/boundary-conditions mass-flow-inlet inlet foam yes no %.10e q\n" %(foams[tech + " " + lg]["rho"]*int(q)/60000000)
            if conditions['outflow'] == True:
                setboundary     += "/define/boundary-conditions/ zone-type outlet-front outflow\n"
                setboundary     += "/define/boundary-conditions/ zone-type outlet-back outflow"
            else:
                setboundary     += "/define/boundary-conditions/ zone-type outlet-front pressure-outlet\n"
                setboundary     += "/define/boundary-conditions/ zone-type outlet-back pressure-outlet\n"
                setboundary     += "/define/boundary-conditions/set pressure-outlet outlet-front outlet-back () mixture p-backflow-spec-gen no yes gauge-pressure no %0.10f q\n" %(dp[q])
            initialize      = "; Initialize the solution\n/solve/initialize/initialize-flow/\n/solve/patch cmc0p02 vessel () mp 1"
            autosave        = "; Set up auto-save\n/file/auto-save data-frequency "+str(saveFrequency)+"\n/file/auto-save/root-name "+ targetDir + caseLoc + "dataFiles/"+fileID+".dat.gz\n/file/auto-save append-file-name-with time-step 6\n/file/cff-files no"
            reportDef       = "; Define report definitions\n"
            reportDef       += "/solve/report-definitions/add surface-coverage surface-areaavg surface-names vein-wall plane-1 () field foam vof per-surface yes q q\n"
            reportDef       += "/solve/report-definitions/add blood-displacement-front flux-massflow zone-names outlet-front () phase cmc0p02 q\n"
            reportDef       += "/solve/report-definitions/add blood-displacement-back flux-massflow zone-names outlet-back () phase cmc0p02 q"
            # reportFileLoc   = "."+ experimentDir.replace("/","\\\\") + caseLoc.replace("/","\\\\") + saveLoc.replace("/","\\\\")
            reportFileLoc   = "./" + saveLoc
            reportFiles     = "; Define report files\n/solve/report-files/add surface-coverage report-defs surface-coverage () frequency "+str(saveFrequency)+" file-name \""+reportFileLoc+ "surface-coverage.out\" q\n/solve/report-files/add blood-displacement-front report-defs blood-displacement-front () frequency "+str(saveFrequency)+" file-name \""+reportFileLoc+ "blood-displacement-front.out\" q\n/solve/report-files/add blood-displacement-back report-defs blood-displacement-back () frequency "+str(saveFrequency)+" file-name \""+reportFileLoc+ "blood-displacement-back.out\" q"
            timeStep        = "; Set time step\n/solve/set/time-step " +str(stepSize)
            transientIt     = "; Set transient dual-time-iterate [number-of-time-steps] [max-number-of-iterations]\n/solve/dual-time-iterate "+ str(nSteps) +" "+ str(maxIterations)
            writeData       = "; Write data file (compressed, iteration number included in file name)\nwd "+ targetDir + caseLoc + "dataFiles/"+fileID +"-00000.dat.gz"
            end             = "; Exit FLUENT\nexit\nyes"

            journal = "; Read case file \nrc "
            journal += targetDir + caseName + "\n\n"
            journal += setmaterial + "\n\n"
            journal += setboundary + "\n\n"
            journal += initialize + "\n\n"
            journal += writeData + "\n\n"
            journal += autosave + "\n\n"
            journal += reportDef + "\n\n"
            journal += reportFiles + "\n\n"
            journal += timeStep + "\n\n"
            journal += transientIt + "\n\n"
            journal += end

            # %% run_fluent Compiler
            if conditions['slurm'] == True:
                run = "#!/bin/bash\n\n"
                run += "#SBATCH --nodes="+str(nodes) + "\n"
                run += "#SBATCH --cpus-per-task=1\n"
                run += "#SBATCH --ntasks-per-node=" + str(ppn) + "\n"
                run += "#SBATCH --ntasks="+str(nodes*ppn) + "\n"
                run += "#SBATCH --time="+walltime[q] + "\n"
                run += "#SBATCH --partition=batch\n\n"
                run += "#Change to directory from which job was submitted\ncd $SLURM_SUBMIT_DIR" +"\n\n"
                run += "#Load fluent module so that we find the fluent command\nmodule load fluent/" + fluent +"\n\n"
                run += "#Run default version of Fluent in 3d mode in parallel over $SLURM_NTASKS processors\n"
                run += "srun hostname -s |sort -V > $(pwd)/slurmhosts.$SLURM_JOB_ID.txt\n\n"
                run += "fluent 3ddp -pinfiniband -mpi=intel -t$SLURM_NTASKS -cnf=slurmhosts.$SLURM_JOB_ID.txt -g -i " + targetDir + caseLoc + journalName
                run += " > " + targetDir + caseLoc + "dataFiles/" + "output_file_$SLURM_JOB_ID"
            else:
                run = "#!/bin/bash\n\n"
                run += "#PBS -l nodes="+str(nodes)+":ppn="+str(ppn) + "\n"
                run += "#PBS -l walltime="+walltime[q] + "\n\n"
                run += "#Change to directory from which job was submitted\ncd $PBS_O_WORKDIR" +"\n\n"
                run += "# set number of processors to run on (using list of node names in file $PBS_NODEFILE)\n"
                run += "nprocs=`wc -l $PBS_NODEFILE | awk '{ print $1 }'`" + "\n\n"
                run += "#Load fluent module so that we find the fluent command\nmodule load fluent/" + fluent +"\n\n"
                run += "#Run default version of Fluent in 3d mode in parallel over $nprocs processors\n"
                run += "fluent 3ddp -rsh -t$nprocs -cnf=$PBS_NODEFILE -g -i " + targetDir + caseLoc + journalName
                run += " > " + targetDir + caseLoc + "dataFiles/" + "output_file_$PBS_JOBID"

            # %% Save Scripts
            pcDir = "/Users/alizz/OneDrive - University of Southampton/Shared Folder - Venous Simulation/SimpleTube (PoC)/IRIDIS Directory upload (SimpleTube)/"
            savePath = pcDir + caseLoc
            if conditions['slurm'] == True:
                submission += "cd " + targetDir + caseLoc + "\n"
                submission += "sbatch " +runName + "\n\n"
            else:
                submission += "cd " +targetDir + caseLoc + "\n"
                submission += "qsub "+ runName + "\n\n"
            if os.path.exists(savePath) == False:
                raise IOError
            with open(os.path.join(savePath + journalName),'w') as fileJou:
                fileJou.write(journal)
            with open(os.path.join(savePath + runName),'w') as fileRun:
                fileRun.write(run)
with open(os.path.join(pcDir,"submission.txt"),'w') as fileQsub:
    fileQsub.write(submission)
get_ipython().magic('clear')  # clear the console
print(journal)
print('\n\n')
print(run)
print('\n\n')
print(submission)