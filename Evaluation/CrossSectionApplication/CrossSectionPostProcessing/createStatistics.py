from Evaluation.CrossSectionApplication.CrossSectionPostProcessing.ThreeDFrontToNetCDF import prepareData
from era5dataset.ERA5Reader.ConversionMethods.variableConversions import equivalentPotentialTempFromQ

import numpy as np
import os




def create1DStatistic(indata, separation_from_file, splitData = None, verbose = False):
    datas = prepareData(indata, verbose) 
   
    smpl_dat = datas[list(datas.keys())[0]][0]

    in_height = 16 
    in_variables = (smpl_dat.shape[0]-2)//in_height
    
    myResult = np.zeros((in_variables+1, in_height,0))
    count = 0
    window_size = 10
    if(separation_from_file):
        # only used when separating (as some 0 files are ignored)
        idx_count = 0
        # ml20160101_00_1.npy --> ml20160101_00_warm.npy
    for i in datas:
        for idx in range(len(datas[i])):
            if(datas[i][idx].shape[-1]>0):
                data = datas[i][idx]
                latlon, valdat = data[-2:], data[:-2].reshape(in_variables,in_height,data.shape[-2],-1)
                ept = equivalentPotentialTempFromQ(t = valdat[0], q = valdat[1], PaPerLevel = valdat[-1]*100)
                valdat = np.concatenate((valdat, ept[None,...]), axis=0)
                variables,height, width, samples = valdat.shape
                if(separation_from_file):
                    # load split from file:
                    split = splitData[0].astype(np.int32)

                    leftMean = np.zeros((variables, height, samples))
                    rightMean = np.zeros((variables, height, samples))
                    for s in range(samples):
                        for h in range(height):
                            leftMin = max(0,split[h,idx_count]-window_size)
                            rightMax = min(width,split[h,idx_count]+window_size)
                            valid = (i in ["cold"] and split[h,idx_count] < 140) or (i in ["warm"] and split[h, idx_count] > 60)
                            if valid:
                                leftMean[:, h, s] = valdat[:,h,leftMin:split[h,idx_count],s].mean(axis=-1)
                                rightMean[:, h, s] = valdat[:,h,split[h,idx_count]+1:rightMax,s].mean(axis=-1)
                            else:
                                leftMean[:,h,s] = np.nan
                                rightMean[:,h,s] = np.nan

                            valdat[:,h,split[h,idx_count],s] = np.max(valdat[:,:,:,s], axis=(1,2))
                        idx_count+=1   
                else:
                    potOffset = valdat.shape[2]//2-window_size
                    # for random split (split randomly)
                    split = width//2+ np.random.random_integers(-potOffset, potOffset, size = (height, samples))
                    # for static split (split always in the middle)
                    #split = width//2+ np.zeros((height, samples), dtype=np.int32)
                    leftMean = np.zeros((variables, height, samples))
                    rightMean = np.zeros((variables, height, samples))
                    for s in range(samples):
                        for h in range(height):
                            leftMin = max(0,split[h,s]-window_size)
                            rightMax = min(width,split[h,s]+window_size)
                            leftMean[:, h, s] = valdat[:,h,leftMin:split[h,s],s].mean(axis=-1)
                            rightMean[:, h, s] = valdat[:,h,split[h,s]+1:rightMax,s].mean(axis=-1)
                meandiff = leftMean-rightMean
                myResult = np.concatenate((myResult,meandiff),axis=-1)
                count += valdat.shape[-1]
    
    return myResult
    

def createStatisticsFromPipeline(data, separation_from_file, inname, tgt_name, split_location):
    data = data
    if(separation_from_file):
        splitf,ext = os.path.splitext(os.path.basename(inname))
        splitf = "_".join(splitf.split("_"))+"_{}".format(tgt_name)+".npy"
        splitdata = np.load(os.path.join(split_location, splitf))
    else:
        splitdata = None
    return createStatisticsFromData("1D", data, separation_from_file, splitdata, False)


def createStatisticsFromData(stat_type, data, separation_from_file, splitlocations, verbose = False):
    if stat_type == "1D":
        return create1DStatistic(data, separation_from_file, splitlocations, verbose)
    else:
        return np.zeros(0)
    









    
    




