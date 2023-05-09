from netCDF4 import Dataset
import h5py
import os
import numpy as np
from scipy.ndimage import map_coordinates


from .ConversionMethods.generalConversions import *
from .ConversionMethods.variableConversions import *
from .ConversionMethods.coordinateConversions import *
from .ConversionMethods.ConversionFormulas import setConversionFormulas
import numexpr as ne
import pandas as pd

class BinaryResultReader:
    def __init__(self):
        pass
    def read(self, filename, latrange, lonrange):
        data = np.fromfile(filename, dtype=np.bool).reshape(720,1480,5)[:,20:-20]
        mylonRange = (np.arange(lonrange[0], lonrange[1], 0.25)*4).astype(np.int32)
        mylatRange = (np.arange(latrange[0], latrange[1], -0.25)*4).astype(np.int32)
        mylonRange += 180*4
        mylatRange = 90*4 - mylatRange
        img = data[mylatRange]
        img = img[:,mylonRange]
        return img


class CDFReader:
    def __init__(self, filetype = 0, normType = None, sharedObj = None, advancedReading = True):
        # 0 -> h5py, 1 -> netCDF4
        self.filetype = filetype
        self.asHDF5 = filetype == 0
        self.setNormalizationType(normType)
        
        # Basic Reading: 
            # Read Values from Disk
            # Possibly interpolate onto a grid if desired
        # Advanced Reading:
            # Basic Reading
            # Apply functions to variables (e.g. pol, abs, delta) or generate composites (e.g. ept, dewpoint, ...) 
                # using cache for faster access
            # Allow for Normalization of values (not implemented)

        self.setAdvancedReading(advancedReading)

        #self.warp_func = np.vectorize(map_coordinates, signature='(b,m),(2,l,n)->(l,n)')
        self.warp_func = warpImage
        if(sharedObj is None):
            self.bufferMasks = False
        else:
            self.bufferMasks = True
            self.cnt = 0
            self.lock = sharedObj[0]
            self.pathToMasks = sharedObj[1]

    def setNormalizationType(self, normalization_type):
        #None -> No normalization, 0 -> min,max , 1 -> mean, var
        self.normalize_type = normalization_type
        self.normalize = not self.normalize_type is None

    def setAdvancedReading(self, status):
        self.advancedReading = status
        if(self.advancedReading):
            self.setPossibleFormulas()
        else:
            self.availableFormulas = None

    def getDataModel(self, filename):
        rootgrp = self.getRootgrp(filename)
        data_model = rootgrp.data_model
        rootgrp.close()
        return data_model
    def __repr__(self):
        myString = "CDFReader\n"
        myString += str(self.__dict__)
        return myString 
    
    def getVars(self, filename):
        rootgrp = self.getRootgrp(filename)
        vars = self.getVarsFromRootgrp(rootgrp)
        rootgrp.close()
        return vars

    def getVarsFromRootgrp(self, rootgrp):
        if(self.asHDF5):
            return np.asarray(rootgrp)
        else:
            return np.array(list(rootgrp.variables.keys()))
    
    def getRootgrp(self, filename):
        if(self.asHDF5):
            return h5py.File(os.path.realpath(filename), "r")
        else:
            return Dataset(os.path.realpath(filename), "r", format="NETCDF4", parallel=False)

    def getLocalDimNames(self, keys):
        xydim = ["longitude", "latitude",None]
        if("longitude" in keys):
            pass
        elif("lon" in keys):
            xydim[0] = "lon"
        if("latitude" in keys):
            pass
        elif("lat" in keys):
            xydim[1] = "lat"
        if("level" in keys):
            xydim[2] = "level"
        elif("plev" in keys):
            xydim[2] = "plev"
        return xydim


    def getDims(self, filename):
        rootgrp = self.getRootgrp(filename)
        keys = self.getVarsFromRootgrp(rootgrp)
        xydim = self.getLocalDimNames(keys)
        if(xydim[2] is None):
            dims = rootgrp[xydim[0]][:],rootgrp[xydim[1]][:],None
        else: 
            dims = rootgrp[xydim[0]][:],rootgrp[xydim[1]][:],rootgrp[xydim[2]][:]
        rootgrp.close()
        return dims

    def getUnits(self, filename, variables = None, levelrange = None):
        rootgrp = self.getRootgrp(filename)
        keys = self.getVarsFromRootgrp(rootgrp)
        local_dim_names = self.getLocalDimNames(keys)
        noLevels = local_dim_names[2] is None
        if(variables is None):
            variables = keys
        local_latrange, local_lonrange, local_levelrange, local_timerange, local_warpmask = getLocalValues(rootgrp, None, None , levelrange, None, None, local_dim_names, None, noLevels)
        units = getUnitsFromRootgrp(rootgrp, variables, len(local_levelrange), local_dim_names, self.asHDF5)
        rootgrp.close()
        return units



    def setPossibleFormulas(self):           
        self.availableFormulas = setConversionFormulas()



    def InferFormula(self, v, av, formulas):
        # variable can be read directly
        if(v in av):
            return v,1,0
        # go through all available formulas and identify a possible one
        if(v in formulas):
            sub_form = formulas.copy()
            myForm = sub_form.pop(v)
            minFormSpeed = 1000
            bestSpeedReads = 1000
            bestSpeedCalcs = 1000
            bestForm = None
            # for each formula + calculations pair (fc)
            for fc in myForm:
                formReads = 0
                formCalcs = fc[1]
                f = fc[0]
                # get the arguments of f
                firstFunc = f.find("(") 
                endFunc = f.rfind(")")
                # no function found, check if variable is directly readable
                # This is always the shortest path possible
                if(firstFunc == -1):
                    tgt, reads, calcs  = self.InferFormula(f, av, sub_form)
                    if(tgt is None):
                        continue
                    formReads += reads
                    formCalcs += calcs
                    formSpeed = formCalcs + formReads
                else:
                    # create a list from it
                    fargs = getFuncArguments(f[firstFunc+1:endFunc])
                    
                    # go through all needed arguments
                    pargs =[]
                    for farg in fargs:
                        # can be inferred ? then formula is here
                        # else none
                        tgt, reads, calcs = self.InferFormula(farg, av, sub_form)
                        formReads += reads
                        formCalcs += calcs
                        pargs.append(tgt)
                        if(tgt is None):
                            break
                        formSpeed = formCalcs + formReads
                        if(formSpeed > minFormSpeed):
                            break
                    # invalid formula => go to next Formula
                    if(None in pargs):
                        continue
                # If the formula is "faster" than the current best, replace it
                if(formSpeed<minFormSpeed):
                    if(firstFunc == -1):
                        bestForm = tgt
                    else:
                        bestForm = "{0}({1})".format(v, ",".join(pargs))
                    minFormSpeed = formSpeed
                    bestSpeedCalcs = formCalcs
                    bestSpeedReads = formReads

            return bestForm, bestSpeedReads, bestSpeedCalcs
        return None, 0, 0

    def stripFunctionArguments(self, v):
        fargsbegin = v.find("(")
        if(fargsbegin != -1):
            return v[:fargsbegin]
        else:
            return v

    def inferVariableTransforms(self, available_variables, desired_variables):
        if(not self.advancedReading):
            print("formula inference not possible!")
            print("Please set advanced reading active, to enable formula inference")
            return desired_variables
        else:
            out = []
            for v in desired_variables:
                v = self.stripFunctionArguments(v)
                poss_form, reads, calcs = self.InferFormula(v, available_variables, self.availableFormulas)
                out.append(poss_form)
            print("result: read", desired_variables, "as", out, "with", reads, "reads and", calcs, "calcs")
            return out
    
    

    # extract available varaibles from filename before inference
    def inferVariableTransformsFromFile(self, filename , desired_variables):
        return self.inferVariableTransforms(self.getVars(filename), desired_variables)

    
    def getQueryInfo(self, filename, variables = None, levelrange = None, asDataFrame = False):
        """Get Information about the content along Axis-0 for a given query

        Keyword arguments:

        filename -- Path to the NetCDF file

        variables -- Variables that should be read, if None all available variables are read (default None)

        levelrange -- List of leves that should be extracted. If None all levels are extracted (default None)

        asDataFrame -- If True the QueryInfo is returned as a DataFrame object with indices ["query", "level", "unit"]. Else the QueryInfo is returned as a Numpy array

        Returns 2D-QueryInfo.

        QueryInfo contains information about the "query", "level" and "unit" for each row of the 
        corresponding query using the readers "read" function. 
        E.g. Querying a datafile that stores temperature in Kelvin as "t" and specific humidity as "q", within a 20x40 grid using the query: 
        variables = ["t", "q"], levelrange = [105,107].
        The resulting data-array from the read function will be of shape [4, 20, 40].
        The corresponding Query-Info will be [["t", 105, "K"],["t", 107, "K"],["q", 105, "kg kg**-1"],["q", 107, "kg kg**-1"]]."""
        rootgrp = self.getRootgrp(filename)
        keys = self.getVarsFromRootgrp(rootgrp)
        local_dim_names = self.getLocalDimNames(keys)
        noLevels = local_dim_names[2] is None
        if(variables is None):
            variables = keys
        local_latrange, local_lonrange, local_levelrange, local_timerange, local_warpmask = getLocalValues(rootgrp, None, None , levelrange, None, None, local_dim_names, None, noLevels)
        if(noLevels):
            myLevels = local_levelrange
        else:
            myLevels = rootgrp[local_dim_names[2]][local_levelrange]
        info = getQueryInfoFromRootgrp(rootgrp, variables, myLevels, local_dim_names, self.asHDF5)
        rootgrp.close()
        if(asDataFrame):
            return pd.DataFrame(info, ["query", "level", "unit"])
        else:
            return info.transpose()
    
    def readWithInfo(self, filename, variables = None, latrange = None, lonrange = None, levelrange = None, lat_step = None, lon_step = None, warpmask = None, timestep = 0, asDataFrame = False):
        '''Read data from file and return it as a 3D array (see read) and additionally provide infromation for each row of the result (see getQueryInfo)'''
        info = self.getQueryInfo(filename, variables, levelrange, asDataFrame)
        data = self.read(filename, variables, latrange, lonrange, levelrange, lat_step, lon_step, warpmask, timestep)
        return data, info
        
        

    def read(self, filename, variables = None, latrange = None, lonrange = None, levelrange = None, lat_step = None, lon_step = None, warpmask = None, timestep = 0):
        """Read variable data from NetCDF file within a requested range to an output grid

        Keyword arguments:

        filename -- Path to the NetCDF file

        variables -- Variables that should be read, if None all available variables are read (default None)
        
        latrange -- Tuple(lat_from, lat_to) that describes the latitude interval "[lat_from, lat_to[" to extract. If None, all latitudes are extracted (default None)
        
        lonrange -- Tuple(lat_from, lat_to) that describes the longitude interval "[lat_from, lat_to[" to extract. If None, all longitudes are extracted (default None)
        
        levelrange -- List of leves that should be extracted. If None all levels are extracted (default None)
        
        lat_step -- Step at which latitudes within latrange should be extracted. Forces the reader to read all values defined by the slice(lat_from, lat_to, lat_step). If None, all latitudes within latrange are extracted. (default None)
        
        lon_step -- Step at which longitudes within lonrange should be extracted.  Forces the reader to read all values defined by the slice(lon_from, lon_to, lon_step). If None, all longitudes within lonrange are extracted. (default None)
        
        warpmask -- Desired output grid in degree. Extracted Values are warped onto the grid using first order interpolation. If None the data will be returned as read. (default None)

        Returns 3D Numpy-array, containing read data from file <filename>

        Axes (1,2) correspond to the extracted latitdue and longitude. 
        Each row (axis 0) corresponds to an entry of a flattened list for Variables and Levels
        For variables = [var_0, var_1, ...,var_n] and levelrange = [level_0, ..., level_n]
        the resulting order of axis 0 will be: [var_0 level_0, var_0 level_1 ,..., var_0 level_n, var_1 level_0 ,..., var_n level_n]
        Note: If Multilevel (e.g. temperature) and Singlelevel (e.g. latitude, longitude) variables are extracted at the same time, all singlelevel variables will be read at the end.
        E.g. variables = [latitude, t, q] will be extracted as if variables were [t, q, latitude]. This eases identifying indices in the resulting array.

        Axis-0-Information can be requested using the readers getQueryInfo() function.
        """
        # Open the file depending on the filetype
        rootgrp = self.getRootgrp(filename)
        
        
        # If no variables are given, we extract all variables in the dataset
        keys = self.getVarsFromRootgrp(rootgrp)
        if(variables is None):
            variables = keys
        # get The dimensions of the file
        local_dim_names = self.getLocalDimNames(keys)
        noLevels = local_dim_names[2] is None
        wherelats = np.where(np.isin(variables, ["lat","latitude"]))[0]
        wherelons = np.where(np.isin(variables, ["lon","longitude"]))[0]
        if(len(wherelats)>0):
            variables[wherelats[0]] = local_dim_names[1]
        if(len(wherelons)>0):
            variables[wherelons[0]] = local_dim_names[0]

        # Set the local values first
        local_latrange, local_lonrange, local_levelrange, local_timerange, local_warpmask = getLocalValues(rootgrp, latrange, lonrange, levelrange, lat_step, lon_step, local_dim_names, warpmask, noLevels)
        # Read the values from the file
        if(self.advancedReading):
            myImage = extractImageFromCDFh5pyChunkedSlim1dAfterNormGeneralDerivativeAndCache(rootgrp, variables, local_latrange, local_lonrange, local_levelrange, local_timerange, local_dim_names, self.asHDF5, noLevels)
        else:
            myImage = extractImageFromCDFh5pyChunked(rootgrp, variables, local_latrange, local_lonrange, local_levelrange, local_timerange, local_dim_names, self.asHDF5, noLevels)
        # Here we can already close the file before post processing
        rootgrp.close()
        
        # warp image if a warp mask is given
        performWarp = not local_warpmask is None
        if(performWarp):
            myImage = self.warp_func(myImage, local_warpmask)
        return myImage
        

def getLocalValues(rootgrp, latrange, lonrange, levelrange , lat_step, lon_step, local_dim_names, warpmask, noLevels):
        local_latrange, local_lonrange, local_levelrange = get_local_ranges(rootgrp, latrange, lonrange, levelrange, lat_step, lon_step, local_dim_names, noLevels)

        # If warping is to be performed, we restrict ourselves to the patch that contains the warping area
        local_warpmask = None
        if(not warpmask is None):
            local_warpmask = get_local_warpmask(rootgrp, warpmask, local_lonrange, local_latrange, local_dim_names)
            
        #TODO --- make this work for time axis as well 
        return local_latrange, local_lonrange, local_levelrange, np.arange(0,1), local_warpmask

def get_local_ranges(rootgrp, latrange, lonrange, levelrange, lat_step, lon_step, local_dim_names, noLevels):
    # factor for innacurate matching at the edges of the lat/lon range intervall, due to inaccuricies in the floating point calculation
    # (e.g. x + 0.02*y = z.99999999  or z.000000001 )
    # this leads to erroneous extractions of various sizes
    epsilon = 0.0001
    
    # no level exists, set a default 1 level (as the output would have)
    if(noLevels):
        if(not levelrange is None):
            print("Warning: Level does not exists for this data, yet it was requested!")
        levelrange = np.ones(1).astype(np.int32)

    # if no levelrange is given, set Levelrange to all levels as default
    if(levelrange is None):
        levelrange = rootgrp[local_dim_names[2]][:].astype(np.int32)

    loc_lat = rootgrp[local_dim_names[1]][:]
    # set latrange to all lats as default
    if(latrange is None):
        val_latrange = loc_lat
    else:
        # adjust latrange a little to compensate for numerical errors
        if(lat_step is None):
            tgt_lat_dir = np.sign(latrange[1]-latrange[0])
            latrange = (latrange[0]-epsilon*tgt_lat_dir, latrange[1]+epsilon*tgt_lat_dir)
            
            if(tgt_lat_dir > 0):
                # sort valid points, to get the increasing values
                val_latrange = np.sort(loc_lat[(loc_lat >= latrange[0]) & (loc_lat <= latrange[1])])
            else:
                # sort valid points, but then reverse the order! to get the decreasing values
                val_latrange = np.sort(loc_lat[(loc_lat <= latrange[0]) & (loc_lat >= latrange[1])])[::-1]
        else:
            latrange = (latrange[0]-epsilon*np.sign(lat_step), latrange[1]+epsilon*np.sign(lat_step))
            val_latrange = np.arange(latrange[0], latrange[1]+0.1*lat_step, lat_step)
    # set lonrange to all lons as default
    loc_lon = rootgrp[local_dim_names[0]][:]
    loc_lon = ne.evaluate('loc_lon%360')
    if(lonrange is None):
        val_lonrange = loc_lon
    else:
        if(lon_step is None):
            tgt_lon_dir = np.sign(lonrange[1]-lonrange[0])
            lonrange = (lonrange[0]-epsilon*tgt_lon_dir, lonrange[1]+epsilon*tgt_lon_dir)
            if(tgt_lon_dir > 0):
                intervals_to_zero = (np.array([lonrange[0]])//360).astype(np.int32)
                # lonrange[1] is exclusive => use SDIV (adjusted for float) instead
                intervals_from_zero = (np.array([lonrange[1]])//360 - ((lonrange[1]%360) == 0)).astype(np.int32)
                intervals = intervals_from_zero-intervals_to_zero
                centers = [lonrange[0]]+[360*(tgt_lon_dir>0)]*(intervals[0])+[lonrange[1]%360]
                # this case has to be handled manually
                if(centers[-1] == 0):
                    centers[-1] = 360
            else:
                intervals_to_zero = (np.array([lonrange[0]])//360 - ((lonrange[0]%360) == 0)).astype(np.int32)
                # lonrange[1] is exclusive => use SDIV (adjusted for float) instead
                intervals_from_zero = (np.array([lonrange[1]])//360).astype(np.int32)
                intervals = -(intervals_from_zero-intervals_to_zero)
                centers = [lonrange[0]%360]+[360]*(intervals[0])+[lonrange[1]]
                if(centers[0] == 0):
                    centers[0] = 360
                    # As modulo does not work well for negative strides, manually add extra values
                    centers = [0] + centers
            val_lonrange=np.zeros(0)
            
            for i in range(len(centers)-1):
                if(tgt_lon_dir > 0):
                    v1 = centers[i]%360
                    v2 = centers[i+1]
                    # sort valid points, to get the increasing values
                    val_lonrange = np.concatenate((val_lonrange, np.sort(loc_lon[(loc_lon >= v1) & (loc_lon <= v2) ])))
                else:
                    v1 = centers[i]
                    v2 = centers[i+1]%360
                    # sort valid points, but then reverse the order! to get the decreasing values
                    val_lonrange = np.concatenate((val_lonrange, np.sort(loc_lon[(loc_lon <= v1) & (loc_lon >= v2) ])[::-1]))
            if(tgt_lon_dir<0 and lonrange[1] in rootgrp[local_dim_names[0]][:]):
                val_lonrange = val_lonrange[:-1]
        else:
            lonrange = (lonrange[0]-epsilon*np.sign(lon_step), lonrange[1]+epsilon*np.sign(lon_step))
            val_lonrange = np.arange(lonrange[0], lonrange[1], lon_step)
            ne.evaluate('val_lonrange%360')
    # all lon values are now between 0 and 360 in the rootgrp
    
    loc_lon_sort = np.argsort(loc_lon)
    local_lonrange1 = np.take(loc_lon_sort, np.searchsorted(loc_lon[loc_lon_sort], val_lonrange))
    
    if(noLevels):
        loc_level = np.ones(1).astype(np.int32)
    else:
        loc_level = rootgrp[local_dim_names[2]][:]
    loc_level_sort = np.argsort(loc_level)
    local_levelrange1 = np.take(loc_level_sort, np.searchsorted(loc_level[loc_level_sort], levelrange))

    loc_lat_sort = np.argsort(loc_lat)
    
    local_latrange1 = np.take(loc_lat_sort, np.searchsorted(loc_lat[loc_lat_sort], val_latrange))
    #Security check
    if(np.any(loc_lat[local_latrange1] != val_latrange)):
        raise Exception("Invalid Latitude specified! Dataset does not contain all requested latitudes!\n \
        Requested Range: {} with step {} \n \
        Calculated Lats: {} \n \
        Available Lats: {}".format(latrange, lat_step, val_latrange, loc_lat))

    if(np.any(loc_lon[local_lonrange1] != val_lonrange)):
        raise Exception("Invalid Longitude specified! Dataset does not contain all requested longitudes!\n \
        Requested Range: {} with step {} \n \
        Calculated Lons (Modulo 360°): {} \n \
        Available Lons (Modulo 360°): {}".format(lonrange, lon_step, val_lonrange, loc_lon))

    if(np.any(loc_level[local_levelrange1] != levelrange)):
        raise Exception("Invalid Level specified! Dataset does not contain all requested levels!\n \
        Requested Levels: {} \n \
        Available Levels: {}".format(levelrange, loc_level))


    if(np.unique(local_latrange1).shape[0] <= 1):
        raise Exception("Invalid Latitude specified! Empty range or dataset does not contain any latitudes within the requested range!!\n \
        Requested Range: {} \n \
        Calculated Lats: {} \n \
        Available Lats: {}".format(latrange, val_latrange, loc_lat))
    if(np.unique(local_lonrange1).shape[0] <= 1):
        raise Exception("Invalid Longitude specified! Empty range or dataset does not contain any longitudes within the requested range!!\n \
        Requested Range: {} \n \
        Calculated Lons (Modulo 360°): {} \n \
        Available Lons (Modulo 360°): {}".format(lonrange, val_lonrange, loc_lon))
    if(local_levelrange1.shape[0] < 1):
        raise Exception("Invalid Level specified! Empty levels!\n \
        Requested Levels: {} \n \
        Available Levels: {}".format(levelrange, loc_level))
    return local_latrange1, local_lonrange1, local_levelrange1

def get_local_warpmask(rootgrp, warpmask, lonrange, latrange, local_dim_names):
    # the grid we want to output (e.g. -180)
    latg, long = warpmask

    latgrid = rootgrp[local_dim_names[1]][:]
    longrid = rootgrp[local_dim_names[0]][:]

    warp_latrange = latgrid[latrange]
    warp_lonrange = longrid[lonrange]


    # latitudinal distance between two grid points of our target grid (orientation of final cut)
    difflat = np.abs(latgrid[1] - latgrid[0])*np.sign(warp_latrange[1]-warp_latrange[0])
    difflon = np.abs(longrid[1] - longrid[0])*np.sign(warp_lonrange[1]-warp_lonrange[0])
    
    #offset of the grid to read from in degree (e.g. 20) (relative to minimum longitude and maximum latitude) (upper left corner)
    offlat = np.max(warp_latrange)
    offlon = np.min(warp_lonrange)

    # offset between the source grid towards the target grid (e.g. -200)
    latg = ne.evaluate('latg-offlat')
    long = ne.evaluate('(long-offlon)%360')

    # bring the degree distance in read grid pixel
    diff = 0
    if(difflat > 0):
        diff = np.abs(warp_latrange[0]-warp_latrange[-1])
    latg = ne.evaluate('(latg+diff)/difflat')

    diff = 0
    if(difflon < 0):
        diff = np.abs(warp_lonrange[0]-warp_lonrange[-1])

    long = ne.evaluate('(long-diff)/difflon')
    local_warpmask = np.array([latg,long])
    return local_warpmask

def get_local_dim_steps(timerange, levelrange, latrange, lonrange):
    local_level_step = 1
    local_lat_step = 1
    local_lon_step = 1
    local_time_step = 1
    if(timerange.shape[0]>1):
        local_time_step = timerange[1] - timerange[0]
    if(levelrange.shape[0]>1):
        local_level_step = levelrange[1]-levelrange[0]
    if(latrange.shape[0]>1):
        local_lat_step = latrange[1]-latrange[0]
    if(lonrange.shape[0]>1):
        # avoid the case, where the first entry is 0 and the second is len(lons)-1
        local_lon_step = np.sign(lonrange[1]-lonrange[0])# if abs(lonrange[1]-lonrange[0]) < abs(lonrange[2]-lonrange[1]) else lonrange[2]-lonrange[1]
    return local_time_step, local_level_step, local_lat_step, local_lon_step

def getChunkSizes(rootgrp, variables, asHDF5):
    #time, level, lat, lon
    chunking = np.array([1,20000,20000,20000])
    useChunks = True
    if(useChunks):
        test_var = variables[0]
        # use the temperature representative 
        if(asHDF5):
            lrootgrp = np.asarray(rootgrp)
        else:
            lrootgrp = np.array(list(rootgrp.variables.keys()))
        if("t" in lrootgrp):
            test_var = "t"
        elif("var129" in lrootgrp):
            test_var = "var129"
        elif("T" in lrootgrp):
            test_var = "T"

        if(asHDF5):
            chunk_shape = rootgrp[test_var].chunks
        else:
            chunk_shape = rootgrp[test_var].chunking()
        if(chunk_shape is not None):
            chunking = chunk_shape[:chunking.shape[0]]
            
    return chunking

def getSourceAndTargetSlices(ranges, chunks, steps):
    batches_from = []
    batches_to = []
    # time, level, lat, lon
    for dimIdx in range(len(ranges)):
        lr2 = len(ranges[dimIdx])
        pos1 = np.where(ranges[dimIdx]%chunks[dimIdx] == 0)[0]
        if(len(pos1) == 0):
            pos1 = np.concatenate(([0], pos1,[lr2]))
        if(pos1[0] != 0):
            pos1 = np.concatenate(([0],pos1))
        if(pos1[-1] != lr2):
            pos1 = np.concatenate((pos1,[lr2]))
        pos1[1:-1-(pos1[-2]==(lr2-1))] += 1 * (steps[dimIdx] < 0)
        batches_from.append([slice(ranges[dimIdx][pos1[i]], ranges[dimIdx][pos1[i+1]-1]+steps[dimIdx] if ranges[dimIdx][pos1[i+1]-1]+steps[dimIdx] >= 0 else None, steps[dimIdx]) for i in range(len(pos1)-1)])
        batches_to.append([slice(pos1[i], pos1[i+1], 1) for i in range(len(pos1)-1)])
    source = np.array([[[[(t,x ,lat,lon) for lat in batches_from[-2]] for lon in batches_from[-1]] for x in batches_from[-3]] for t in batches_from[-4]])
    target = np.array([[[[(x ,lat,lon) for lat in batches_to[-2]] for lon in batches_to[-1]] for x in batches_to[-3]] for t in batches_to[-4]])
    return source[0], target[0]

def getQueryInfoFromRootgrp(rootgrp, variables, levelrange, dim_names, asHDF5):
    levels = len(levelrange)
    listOfDims = [dim_names[0], dim_names[1],'kmPerLon','kmPerLat','time']
    listOfDims += ["cache({})".format(x) for x in listOfDims]
    singleLevelVariables= np.nonzero(np.isin(variables,listOfDims, True))[0].shape[0]
    myUnits = np.empty((len(variables)-singleLevelVariables)*levels+singleLevelVariables , dtype = np.object_)    
    myQuery = np.empty((len(variables)-singleLevelVariables)*levels+singleLevelVariables , dtype = np.object_)  
    myLevel = np.empty((len(variables)-singleLevelVariables)*levels+singleLevelVariables , dtype = np.int32)  
    idx = 0
    svalIdx = 0
    # For each variables
    for full_variable in variables:
        if(full_variable in ["time"]):
            continue
        
        # get Index Range 
        idxRange = slice(idx*levels,(idx+1)*levels,1)
        idx+=1
        # correct index range for single level variables (dimensions)
    
        if(full_variable in listOfDims):
            idxRange = -1 - svalIdx
            svalIdx += 1
            # Special case, always at the last slot
            idx -= 1
            myLevel[idxRange] = 0
        else:
            myLevel[idxRange] = levelrange        
        # get The info
        myUnits[idxRange] = ""
        if(asHDF5):
            if(full_variable in np.asarray(rootgrp)):
                if("units" in rootgrp[full_variable].attrs):
                    myUnits[idxRange] = str(rootgrp[full_variable].attrs["units"], "utf-8")
        else:
            if(full_variable in (list(rootgrp.variables.keys()))):
                if "units" in rootgrp[full_variable].__dict__:
                    myUnits[idxRange] = rootgrp[full_variable].units
        myQuery[idxRange] = full_variable

    return np.array([myQuery, myLevel, myUnits])

def getUnitsFromRootgrp(rootgrp, variables, levels, dim_names, asHDF5):
    listOfDims = [dim_names[0], dim_names[1],'kmPerLon','kmPerLat','time']
    listOfDims += ["cache({})".format(x) for x in listOfDims]
    singleLevelVariables= np.nonzero(np.isin(variables,listOfDims, True))[0].shape[0]
    mydat = np.empty((len(variables)-singleLevelVariables)*levels+singleLevelVariables, dtype = np.object_)    
    idx = 0
    svalIdx = 0
    for full_variable in variables:
        if(full_variable in ["time"]):
            continue
        
        # get Index Range 
        idxRange = slice(idx*levels,(idx+1)*levels,1)
        idx+=1
    
        if(full_variable in listOfDims):
            idxRange = -1 - svalIdx
            svalIdx += 1
            # Special case, always at the last slot
            idx -= 1
        
       
        # read the variable
        mydat[idxRange] = ""
        if(asHDF5):
            if(full_variable in np.asarray(rootgrp)):
                if "units" in rootgrp[full_variable].attrs:
                    mydat[idxRange] = str(rootgrp[full_variable].attrs["units"], "utf-8")    
        else:
            if(full_variable in (list(rootgrp.variables.keys()))):
                if "units" in rootgrp[full_variable].__dict__:
                    mydat[idxRange] = rootgrp[full_variable].units
            
    return mydat

def extractImageFromCDFh5pyChunked(rootgrp, variables , latrange , lonrange, levelrange, timerange, dim_names, asHDF5, noLevels = False):
    levels = len(levelrange)
    ranges = (timerange, levelrange, latrange, lonrange)
    steps = get_local_dim_steps(*ranges)
    listOfDims = [dim_names[0], dim_names[1],'kmPerLon','kmPerLat','time']
    listOfDims += ["cache({})".format(x) for x in listOfDims]
    singleLevelVariables= np.nonzero(np.isin(variables,listOfDims, True))[0].shape[0]
    mydat = np.zeros(((len(variables)-singleLevelVariables)*levels+singleLevelVariables,len(latrange),len(lonrange)))    
    #TODO --- make this work for time axis as well 
    if(noLevels):
        chunks = 1,20000,20000,20000
    else:    
        chunks = getChunkSizes(rootgrp, variables, asHDF5)
    source, target = getSourceAndTargetSlices(ranges, chunks, steps)
    
    # Read all requested variables
    idx = 0
    # index for the single level values
    svalIdx = 0
    
    for full_variable in variables:
        
        # get Index Range 
        idxRange = slice(idx*levels,(idx+1)*levels,1)
        idx+=1
    
        if(full_variable in [dim_names[0], dim_names[1], 'time']):
            idxRange = slice(mydat.shape[0]-svalIdx-1,mydat.shape[0]-svalIdx,1)
            svalIdx += 1
            # Special case, always at the last slot
            idx -= 1
        
        # read the variable
        if(asHDF5):
            getVariable(rootgrp, full_variable, mydat[idxRange], source, target, dim_names, noLevels = noLevels)
        else:
            getVariableCDF(rootgrp, full_variable, mydat[idxRange], source, target, dim_names, noLevels = noLevels)

    return mydat

def extractImageFromCDFh5pyChunkedSlim1dAfterNormGeneralDerivativeAndCache(rootgrp, variables , latrange , lonrange, levelrange, timerange, dim_names, asHDF5, noLevels = False):
    levels = len(levelrange)
    ranges = (timerange, levelrange, latrange, lonrange)
    steps = get_local_dim_steps(*ranges)
    listOfDims = [dim_names[0], dim_names[1],'kmPerLon','kmPerLat','time']
    listOfDims += ["cache({})".format(x) for x in listOfDims]
    singleLevelVariables= np.nonzero(np.isin(variables,listOfDims, True))[0].shape[0]
    mydat = np.zeros(((len(variables)-singleLevelVariables)*levels+singleLevelVariables,len(latrange),len(lonrange)))    
    #TODO --- make this work for time axis as well 
    if(noLevels):
        chunks = 1,20000,20000,20000
    else:
        chunks = getChunkSizes(rootgrp, variables, asHDF5)
    
    
    source, target = getSourceAndTargetSlices(ranges, chunks, steps)

    # Read all requested variables
    idx = 0
    # index for the single level values
    svalIdx = 0
    cache = {}
    cached = {}
    for full_variable in variables:
        # get Index Range 
        idxRange = slice(idx*levels,(idx+1)*levels,1)
        idx+=1
    
        if(full_variable in listOfDims):
            idxRange = slice(mydat.shape[0]-svalIdx-1,mydat.shape[0]-svalIdx,1)
            svalIdx += 1
            # Special case, always at the last slot
            idx -= 1
        
        # read the variable
        readVariable(rootgrp, full_variable, mydat[idxRange], source, target, cache, cached, True, dim_names, asHDF5, noLevels, False)
    return mydat


def readVariable(rootgrp, full_variable, mydat, source, target, cache, cached, writeVar, dim_names, asHDF5, noLevels , cacheThis = False):    
    
    if(cacheThis):
        if(full_variable in [dim_names[0],dim_names[1],'kmPerLon','kmPerLat','time']):
            cache[full_variable] = np.zeros_like(mydat[:,0])
        else:
            cache[full_variable] = np.zeros_like(mydat)
        cached[full_variable] = False
    if(full_variable.isnumeric()):
        return float(full_variable)
    # extract the extraction information from the variable string
    # find if a transformation function is used "(" ")"
    firstFunc = full_variable.find("(") 
    endFunc = full_variable.rfind(")")
    fname = full_variable[:firstFunc]

    
    myFuncList = ["pol2d", "abs2d", "delta_u", "delta_v", "ept", "dewp", "q", "r"]
    # this ensure dot before point
    if("**" in full_variable):
        full_variable = "#".join(full_variable.split("**"))
    operators = ["+","-","*","/","#"]
    # first check for operators (as these are implicitly functions enclosing the current level)
    for o in operators:
        if o in full_variable:
            targetVars = getFuncArguments(full_variable, o)
            buffs = []
            if(len(targetVars)>1):
                for var in targetVars:
                    buffs.append(readVariable(rootgrp, var, mydat, source, target, cache, cached, False, dim_names, asHDF5, noLevels, False))
                for smpl in buffs[1:]:
                    if(o == "#"):
                        o = "**"
                    buffs[0] = eval("buffs[0] {} smpl".format(o))
                if(writeVar):
                    mydat[:] = buffs[0]
                return buffs[0]
    # check for functions
    # If we find a (, that means we have a function to evaluate
    if(firstFunc!=-1):
        # prepare potential buffer arrays
        buffs = []
        # get all variables for function evaluation
        targetVars = getFuncArguments(full_variable[firstFunc+1:endFunc])
        # base is a special case, where we read and write the variable but do not perform any kind of normalization later on
        # We do not need the temporal buffer for those
        if(fname == "base"):
            return readVariable(rootgrp, targetVars[0], mydat, source, target, cache, cached, writeVar, dim_names, asHDF5, noLevels, False)
        # recursively read those 
        if fname == "cache":
            return readVariable(rootgrp, targetVars[0], mydat, source, target, cache, cached, writeVar, dim_names, asHDF5, noLevels, True)


        for var in targetVars:
            buffs.append(readVariable(rootgrp, var, mydat, source, target, cache, cached, False, dim_names, asHDF5, noLevels, False))
        # BEGIN GENERAL CONVERSIONS
        if(fname in myFuncList):
            evalFunc(mydat, writeVar, buffs, targetVars, fname)
        elif(len(fname) > 0):
            return evalNPFunc(mydat, writeVar, buffs, fname)
        else:
            if(writeVar):
                mydat = buffs[0]
            return buffs[0]
    else:
        if(asHDF5):
            return getVariable(rootgrp, full_variable, mydat, source, target, dim_names, cache, cached, cacheThis, writeVar, noLevels=noLevels)
        else:
            return getVariableCDF(rootgrp, full_variable, mydat, source, target, dim_names, cache, cached, cacheThis, writeVar, noLevels=noLevels)



def evalFunc(mydat, writeVar, buffs, targetVars, fname):
    evalstring = "{}(mydat, False, buffs, targetVars)".format(fname)
    print("evaluating: {}".format(evalstring))
    buffs[0] = eval(evalstring)
    if(writeVar):
        mydat[:] = buffs[0]
    return buffs[0]


def evalNPFunc(mydat, writeVar, buffs, fname):
    evalstring = "np.{}(".format(fname)
    for i in range(len(buffs)):
        evalstring += "buffs[{}]".format(i)
        if(i < len(buffs)-1):
            evalstring += ","
        else:
            evalstring += ")"
    print("evaluating: {}".format(evalstring))
    buffs[0] = eval(evalstring)
    if(writeVar):
        mydat[:] = buffs[0]
    return buffs[0]

def getFuncArguments(funcString, sep = ","):
    targetVars =[]
    cStart = 0
    cEnd = 0
    lCount = 0
    for c in funcString:
        if(c == "("):
            lCount += 1
        elif(c == ")"):
            lCount -= 1
        elif(c == sep and lCount==0):
            targetVars.append(funcString[cStart:cEnd].strip())
            cStart=cEnd+1
        cEnd += 1
    targetVars.append(funcString[cStart:cEnd].strip())
    return targetVars



        
def getVariable(rootgrp, variable, mydat, source, target, dim_names, cache = None, cached = None, cacheThis = False, writeVar = True, noLevels = False):
    if((not cache is None) and variable in cached and cached[variable]):
        if(writeVar):
            mydat = cache[variable]
        return cache[variable]
    else:
        # if either is given, no levels are assumed
        noLevels = noLevels or (dim_names[2] is None)
        # variable shall be written, then the buffer is the target destination
        if(writeVar):
            tgtBuffer = mydat
        # variable shall not be written, then the buffer is a copy of the target destination
        else:
            tgtBuffer = np.zeros_like(mydat)
    
        add_off = 0
        scal_fac = 1
        fillValue = np.nan
        missingValue = np.nan
        if(variable == 'ept' or variable == 'dewt'):
            add_off = (rootgrp['t'].attrs['add_offset'],rootgrp['q'].attrs['add_offset'], rootgrp['sp'].attrs['add_offset'])
            scal_fac = (rootgrp['t'].attrs['scale_factor'],rootgrp['q'].attrs['scale_factor'], rootgrp['sp'].attrs['scale_factor'])
            missingValue = (rootgrp['t'].attrs['missing_value'],rootgrp['q'].attrs['missing_value'], rootgrp['sp'].attrs['missing_value'])
            fillValue = (rootgrp['t'].attrs['_FillValue'],rootgrp['q'].attrs['_FillValue'], rootgrp['sp'].attrs['_FillValue'])
        elif(variable in [dim_names[0], dim_names[1], dim_names[2], "kmPerLon", "kmPerLat"]):
            pass
        else:
            attrs = rootgrp[variable].attrs
            if("add_offset" in attrs):
                add_off = attrs['add_offset']
            if("scale_factor" in attrs):
                scal_fac = attrs['scale_factor']
            if("missing_value" in attrs):
                missingValue = attrs['missing_value']
            if("_FillValue" in attrs):
                fillValue = attrs['_FillValue']
                

    
        
        # Read special variable latitude
        if(variable in [dim_names[1], 'kmPerLon']):
            tmpBuffer = np.array([])
            for lat_batch in range(source.shape[2]):
                # netCDF does not allow negative step in indexing
                tmpBuffer = np.concatenate((tmpBuffer, rootgrp[dim_names[1]][:][source[0,0,lat_batch][2]]))
            if(variable == 'kmPerLon'):
                tmpBuffer = np.abs(tmpBuffer)
                tmpBuffer = np.reshape(LatTokmPerLon(tmpBuffer), (tmpBuffer.shape[0], 1))
                tmpBuffer = np.clip(tmpBuffer, 0.1, 30)/27.7762
            else:
                tmpBuffer = np.reshape(tmpBuffer, (tmpBuffer.shape[0], 1))
            tgtBuffer[:] = np.broadcast_to(tmpBuffer, tgtBuffer.shape)
        elif(variable in [dim_names[0]]):
            tmpBuffer = np.array([])
            for lon_batch in range(source.shape[1]):
                # netCDF does not allow negative step in indexing
                tmpBuffer = np.concatenate((tmpBuffer, rootgrp[dim_names[0]][:][source[0,lon_batch,0][3]]))
            
            tmpBuffer = np.reshape(tmpBuffer, (1, tmpBuffer.shape[0]))
            tgtBuffer[:] = np.broadcast_to(tmpBuffer, tgtBuffer.shape)
        elif(variable in ['plev','level']):
            tmpBuffer = np.array([])
            for level_batch in range(source.shape[0]):
                # netCDF does not allow negative step in indexing
                tmpBuffer = np.concatenate((tmpBuffer, rootgrp[dim_names[2]][:][source[level_batch,0,0][1]]))
            
            tmpBuffer = np.reshape(tmpBuffer, (tmpBuffer.shape[0],1,1))
            tgtBuffer[:] = np.broadcast_to(tmpBuffer, tgtBuffer.shape)
        # there is no distortion in latitudinal direction
        elif(variable == 'kmPerLat'):
            tgtBuffer = np.ones(tgtBuffer.shape)
        elif(variable in ['time']):
            tgtBuffer[:] = rootgrp['time']
        elif((not noLevels) and variable == 'sp'):
            for lon_batch in range(source.shape[1]):
                for lat_batch in range(source.shape[2]):
                    lsource = (0, *source[0,lon_batch,lat_batch][-2:])
                    ltarget = (-1, *target[0,lon_batch,lat_batch][-2:])
                    readHDF5VariablePlain(rootgrp[variable], tgtBuffer, lsource, ltarget)
                    # scale the read values to their true range
            scaleArray(tgtBuffer, scal_fac, add_off)
            filterMissingAndFill(tgtBuffer, fillValue, missingValue)
            
            levelrange = rootgrp[dim_names[2]][slice(source[0,0,0][1].start, source[-1,0,0][1].stop,source[0,0,0][1].step)].astype(np.int32)
            calculatePressureFromML(tgtBuffer, levelrange)

        # Read in a standard variable
        else:
            for level_batch in range(source.shape[0]):
                for lon_batch in range(source.shape[1]):
                    for lat_batch in range(source.shape[2]):
                        if(noLevels):
                            readHDF5VariablePlain(rootgrp[variable], tgtBuffer, (source[level_batch,lon_batch,lat_batch][0],*source[level_batch,lon_batch,lat_batch][-2:]), (target[level_batch,lon_batch,lat_batch][0], *target[level_batch,lon_batch,lat_batch][-2:]))
                        else:
                            readHDF5VariablePlain(rootgrp[variable], tgtBuffer, source[level_batch,lon_batch,lat_batch], target[level_batch,lon_batch,lat_batch])
            
            # scale the read values to their true range
            scaleArray(tgtBuffer, scal_fac, add_off)
            filterMissingAndFill(tgtBuffer, fillValue, missingValue)

        if(cacheThis):
            cache[variable] = tgtBuffer
            cached[variable] = True
        return tgtBuffer

def getVariableCDF(rootgrp, variable, mydat, source, target, dim_names, cache = {}, cached ={}, cacheThis = False, writeVar = True, noLevels = False):
        if(variable in cached and cached[variable]):
            if(writeVar):
                mydat = cache[variable]
            return cache[variable]
        else:
            # if either is given, no levels are assumed
            noLevels = noLevels and (dim_names[2] is None)
            # variable shall be written, then the buffer is the target destination
            if(writeVar):
                tgtBuffer = mydat
            # variable shall not be written, then the buffer is a copy of the target destination
            else:
                tgtBuffer = np.zeros_like(mydat)

        
            # Read special variable latitude
            if(variable in ['latitude', 'lat', 'kmPerLon']):
                tmpBuffer = np.array([])
                for lat_batch in range(source.shape[2]):
                    tmpBuffer = np.concatenate((tmpBuffer, rootgrp[dim_names[1]][source[0,0,lat_batch][2]]))
                if(variable == 'kmPerLon'):
                    tmpBuffer = np.abs(tmpBuffer)
                    tmpBuffer = np.reshape(LatTokmPerLon(tmpBuffer), (tmpBuffer.shape[0], 1))
                    tmpBuffer = np.clip(tmpBuffer, 0.1, 30)/27.7762
                else:
                    tmpBuffer = np.reshape(tmpBuffer, (tmpBuffer.shape[0], 1))
                tgtBuffer[:] = np.broadcast_to(tmpBuffer, tgtBuffer.shape)
            elif(variable in ['longitude', 'lon']):
                tmpBuffer = np.array([])
                for lon_batch in range(source.shape[1]):
                    tmpBuffer = np.concatenate((tmpBuffer, rootgrp[dim_names[0]][source[0,lon_batch,0][3]]))
                tmpBuffer = np.reshape(tmpBuffer, (1, tmpBuffer.shape[0]))
                tgtBuffer[:] = np.broadcast_to(tmpBuffer, tgtBuffer.shape)
            elif(variable in ['plev','level']):
                tmpBuffer = np.array([])
                for level_batch in range(source.shape[0]):
                    # netCDF does not allow negative step in indexing
                    tmpBuffer = np.concatenate((tmpBuffer, rootgrp[dim_names[2]][:][source[level_batch,0,0][1]]))
                
                
                tmpBuffer = np.reshape(tmpBuffer, (tmpBuffer.shape[0],1,1))
                tgtBuffer[:] = np.broadcast_to(tmpBuffer, tgtBuffer.shape)
            # there is no distortion in latitudinal direction
            elif(variable == 'kmPerLat'):
                tgtBuffer = np.ones(tgtBuffer.shape)
            elif(variable in ['time']):
                tgtBuffer[:] = rootgrp['time'][:]
            elif((not noLevels) and variable == 'sp'):
                for lon_batch in range(source.shape[1]):
                    for lat_batch in range(source.shape[2]):
                        lsource = (0, *source[0,lon_batch,lat_batch][-2:])
                        ltarget = (-1, *target[0,lon_batch,lat_batch][-2:])
                        readCDFVariable(rootgrp[variable], tgtBuffer, lsource, ltarget)
                        # scale the read values to their true range
                levelrange = rootgrp[dim_names[2]][slice(source[0,0,0][1].start, source[-1,0,0][1].stop,source[0,0,0][1].step)].astype(np.int32)
                calculatePressureFromML(tgtBuffer, levelrange)

            # Read in a standard variable
            else:
                for level_batch in range(source.shape[0]):
                    for lon_batch in range(source.shape[1]):
                        for lat_batch in range(source.shape[2]):
                            if noLevels:
                                readCDFVariable(rootgrp[variable], tgtBuffer, (source[level_batch,lon_batch,lat_batch][0],*source[level_batch,lon_batch,lat_batch][-2:]), (target[level_batch,lon_batch,lat_batch][0], *target[level_batch,lon_batch,lat_batch][-2:]))
                            else:
                                readCDFVariable(rootgrp[variable], tgtBuffer, source[level_batch,lon_batch,lat_batch], target[level_batch,lon_batch,lat_batch])

            if(cacheThis):
                cache[variable] = tgtBuffer
                cached[variable] = True
            return tgtBuffer

def warpImage(img, mask):
    out = np.zeros((img.shape[0], *mask.shape[1:]))
    for level in range(img.shape[0]):
        map_coordinates(img[level], mask, out[level], order = 1, mode='constant', cval = np.nan)
    return out

def readCDFVariable(in_buffer, out_buffer, in_slice, out_slice):
    if(np.ma.is_masked(in_buffer[tuple(in_slice)])):
        out_buffer[tuple(out_slice)] = in_buffer[tuple(in_slice)].filled(0)
    else:
        out_buffer[tuple(out_slice)] = in_buffer[tuple(in_slice)]

def readHDF5VariablePlain(dataset_in,array_out, in_slice, out_slice):
    a = list(in_slice)
    rev = [False]*(len(a)-1)
    for myidx in range(1,len(a)):
        if a[myidx].step == -1:
            rev[myidx-1] = True
            beg = 0 if a[myidx].stop is None else a[myidx].stop +1
            end = a[myidx].start +1
            a[myidx] = slice(beg, end,1)
    
    dataset_in.read_direct(array_out, tuple(a), tuple(out_slice))
    # The first entry is time, which we do not extract, the second may be omitted in noLevel files
    # => go backwards, as those are always necessary
    for x in range(1,len(a)):
        if rev[-x]:
            array_out[tuple(out_slice)]= np.flip(array_out[tuple(out_slice)], axis=-x)


def scaleArray(data, scal_fac, add_off):
    data *= scal_fac
    data += add_off

def filterMissingAndFill(data, fillValue, missingValue):
    data[(data == fillValue) | (data == missingValue)] = np.nan


