import os
from datetime import datetime, timedelta
import numpy as np


# Group Output Channels together
def filterChannels(data, available_labels, target_labels):     
    reordered_img = np.zeros((1,data.shape[1],data.shape[2],len(target_labels)+1))
    for idx, possLab in enumerate(available_labels, 1):
        isIn = False
        for lidx,labelGroup in enumerate(target_labels,1):
            if(possLab in labelGroup):
                isIn = True
                reordered_img[0,:,:,lidx] += data[0,:,:,idx]
        if(not isIn):
            data[0,:,:,0] -= data[0,:,:,idx]
    reordered_img[0,:,:,0] = data[0,:,:,0]
    return reordered_img


# Transfer a timestamp name from one format to another (keeping folder structure intact)
def getCorrespondingName(in_name, in_format, out_format, time_diff=0):
    basename = os.path.basename(in_name)
    dirname = os.path.dirname(in_name)
    labtime = datetime.strptime(basename, in_format)
    datatime = labtime-timedelta(hours=time_diff)
    dataname = datatime.strftime(out_format)
    dataname = os.path.join(dirname, dataname)
    return dataname
