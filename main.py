from Evaluation.PipelineEvaluator import PipelineEvaluator, EvaluationPipeline
from Evaluation.CrossSectionApplication.CrossSectionCreator import CrossSectionExtractor
from Evaluation.CrossSectionApplication.CrossSectionPostProcessing.ThreeDFrontToNetCDF import prepareAndConvertData
from Evaluation.CrossSectionApplication.CrossSectionPostProcessing.createStatistics import createStatisticsFromPipeline

from era5dataset.ERA5Reader.readNetCDF import BinaryResultReader

from argparse import ArgumentParser

import numpy as np
import os
import datetime as dt


def parse_args():
    parser = ArgumentParser("3D Fronts")
    parser.add_argument("--frontal_data", type=str, help="folder containing the front data")
    parser.add_argument("--file_format", default="ml%Y%m%d_%H.bin", type=str, help="file format of files to read. Used to automatically find corresponding files by timestamp")
    parser.add_argument("--outname", default=".", type=str, help="folder to write to")
    parser = CrossSectionExtractor.add_processor_specific_args(parser)
    return parser.parse_args()


class Dataset:
    def __init__(self, folder, format):
        self.data_fold = folder
        self.dataFormat = format

def validFormat(filename, format):
    try:
        dt.datetime.strptime(filename, format)
        return True
    except:
        return False


def run():

    args = parse_args()

    # assumes frontal data saved in a boolean binary file with a shape of (720, 1480, x). 
    # dim[0] is latitude from 90 to -90 °N, dim[1] is longitude from -185 to 185 °E, dim[2] is channel
    reader = BinaryResultReader()

    latrange = (90, 0)
    lonrange = (-90, 50)

    dataset = Dataset(args.frontal_data, args.file_format)
    files = sorted([os.path.join(args.frontal_data,x) for x in os.listdir(args.frontal_data) if validFormat(x, args.file_format)])
    files = files[:3]
    pipeline = PipelineEvaluator([CrossSectionExtractor.load_from_current_state(dataset, args),
                                  prepareAndConvertData])

    for fileidx in range(len(files)):
        fronts_filename = files[fileidx]

        fronts_data = reader.read(fronts_filename, latrange, lonrange).astype(np.int32)
        info = [{"filename": os.path.basename(fronts_filename), "lat_range":latrange, "lon_range":lonrange, "resolution":(-0.25, 0.25)}]
        pipeline(fronts_data, (None, None, info, None), fileidx, fileidx)
    
    args.splitLocation = os.path.join(args.outname,"Cross_Sections")
    if not os.path.exists(os.path.join(args.outname,"MyResults")):
        os.mkdir(os.path.join(args.outname, "MyResults"))
    args.splitType = "algorithm"
    evalPipeline = EvaluationPipeline([CrossSectionExtractor.load_from_current_state(dataset, args),
                                       createStatisticsFromPipeline], args)
    for fileidx in range(len(files)):
        fronts_filename = files[fileidx]

        fronts_data = reader.read(fronts_filename, latrange, lonrange).astype(np.int32)
        info = [{"filename": os.path.basename(fronts_filename), "lat_range":latrange, "lon_range":lonrange, "resolution":(-0.25, 0.25)}]
        evalPipeline(fronts_data, (None, None, info, None), fileidx, fileidx)
    evalPipeline.finalize()
    
    


if __name__ == "__main__":
    run()




