import numpy as np

from era5dataset.ERA5Reader.ConversionMethods.coordinateConversions import calculatePressureFromML

from .ERA5Reader.readNetCDF import CDFReader
from Util.util import getCorrespondingName


# EraExtractors are special read operations grouped in a class
# They can be used to draw multiple samples using some fixed parameters
# (e.g. same variables and levels over all samples)
# Further they can also combine multiple reads from different files
# (e.g. reading a multilevel and a surface file together)

class DefaultEraExtractor():
    # Determine which variables should be extracted
    def __init__(self, variables = ['t','q','u','v','w'], levels = None, normType = 0, sharedObj= None, ftype=0, constantFile = None, constantVars = None):
        self.setVariables(variables)
        self.setLevels(levels)
        # Create a CDF Reader using h5py and normType normalization
        self.reader = CDFReader(ftype, normType = normType, sharedObj = sharedObj)
        self.constantFile = constantFile
        self.constantVars = constantVars
    def setVariables(self, variables):
        self.variables = variables
    def setLevels(self, levels):
        self.levels = levels
    def setNormType(self, normType):
        self.reader.setNormalizationType(normType)
    def getReader(self):
        return self.reader
    
    def appendConstant(self, data, latrange, lonrange, warpmask):
        # append constant data
        const_data = self.reader.read(self.constantFile, self.constantVars, latrange, lonrange, warpmask = warpmask)
        data = np.concatenate((data,const_data), axis = 0)
        return data


    def __call__(self, filename, latrange, lonrange, seed = 0, warpmask = None):
        if isinstance(filename, list):
            print("Currently not correct!")
            exit(1)
            return self.reader.read(filename, self.variables, latrange,lonrange, self.levels)
        data = self.reader.read(filename, self.variables, latrange,lonrange, self.levels, warpmask = warpmask)
        if(not self.constantFile is None):
            data = self.appendConstant(data, latrange, lonrange, warpmask)
        return data

# Allows reading ERA5 data from a multilevel and single level file in one call
# This does not support calculations that use variables from both files. These need to
# be done manually afterwards
class MultiFileEraExtractor(DefaultEraExtractor):
    def __init__(self, variables=['t', 'q', 'u', 'v', 'w'], levels=None, normType=0, sharedObj=None, ftype=0, constantFile=None, constantVars=None, multi_level_fileformat = "ml%Y%m%d_%H.nc", single_level_fileformat = "B%Y%m%d_%H.nc"):
        super().__init__(variables, levels, normType, sharedObj, ftype, constantFile, constantVars)
        # For some single Level variables reading from a second single Level file may be necessary
        # this way files do not need to be merged
        self.multiLevelFileIdentifier = multi_level_fileformat
        self.singleLevelFileIdentifier = single_level_fileformat
        self.extendSingleLevelPressureToMultiLevel = True

    def __call__(self, filename, latrange, lonrange, seed = 0, warpmask = None):
        if isinstance(filename, list):
            print("Currently not implemented!")
            exit(1)
        # for now only sp is allowed
        # separate variables 
        mlFileVars = self.reader.getVars(filename)
        multiLevelVars = [x for x in self.variables if x in mlFileVars or x in ["lat", "latitude", "lon", "longitude", "kmPerLon"]]
        singleLevelVars = [x for x in self.variables if x not in multiLevelVars]
        
        # read multilevel data
        data = self.reader.read(filename, multiLevelVars, latrange,lonrange, self.levels, warpmask = warpmask)
        # if no single Level Var is identified, skip trying read a second file
        if(len(singleLevelVars) > 0):
            # get single file data
            singleLevelFilename = getCorrespondingName(filename, self.multiLevelFileIdentifier, self.singleLevelFileIdentifier,0)
            currentNormStatus = self.reader.normalize
            self.reader.normalize=False
            singleLevel_data = self.reader.read(singleLevelFilename, singleLevelVars, latrange, lonrange, None, warpmask = warpmask)
            self.reader.normalize=currentNormStatus
            # extend if desired
            levels = self.levels
            if(levels is None):
                levels = self.reader.getDims(filename)[-1]

            if(singleLevel_data.shape[0] > 0):
                if(self.extendSingleLevelPressureToMultiLevel):
                    singleLevel_data = singleLevel_data.repeat(len(levels), axis=0)
                    calculatePressureFromML(singleLevel_data, levels)
            
                # concat
                for k in range(len(singleLevelVars)):
                    w = len(levels) if self.extendSingleLevelPressureToMultiLevel else 1
                    idx = self.variables.index(singleLevelVars[k])
                    data = np.concatenate((data[:len(levels)*idx], singleLevel_data[k*w:(k+1)*w], data[len(levels)*idx:]))
            
        if(not self.constantFile is None):
            data = self.appendConstant(data, latrange, lonrange, warpmask)
        return data

