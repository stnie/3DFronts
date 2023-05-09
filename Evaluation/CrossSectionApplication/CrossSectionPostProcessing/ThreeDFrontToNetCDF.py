
import numpy as np
import datetime as dt
import os

from .ThreeDFrontUtils import find3DSeparation
from Util.IO.Data2NetCDF import netCDFWriter
from era5dataset.ERA5Reader.ConversionMethods.variableConversions import equivalentPotentialTempFromQ


def drawSeparationToData(sep, data3d, data2d, lat, lon, debug = False):
    height, samples = sep.shape
    origsep = sep.copy()
    sep = sep.astype(np.int32)
    for s in range(samples):
        for h in range(height):
            if(not np.isnan(origsep[h,s])):
                if(sep[h,s] <= 0 or sep[h,s] >= lat.shape[0]-1):
                    print(sep[h,s], h, s)
                tgt_lat_pos = lat[sep[h,s]:sep[h,s]+1,s]
                tgt_lon_pos = lon[sep[h,s]:sep[h,s]+1,s]
                if(debug):
                    if s == samples//2:
                        tgt_lat_pos = lat[sep[h,s]-10:sep[h,s]+10,s]
                        tgt_lon_pos = lon[sep[h,s]-10:sep[h,s]+10,s]

                tgt_lat_pos = tgt_lat_pos.astype(np.int32)
                tgt_lon_pos = tgt_lon_pos.astype(np.int32)

                validIndices = (tgt_lat_pos>=0) * (tgt_lat_pos < data3d.shape[2])*(tgt_lon_pos>=0) * (tgt_lon_pos < data3d.shape[3])
                tgt_lat_pos =  tgt_lat_pos[validIndices]
                tgt_lon_pos =  tgt_lon_pos[validIndices]

                data3d[0,h,tgt_lat_pos, tgt_lon_pos] = 1
                data2d[0,h,s] = sep[h,s]


def addAllCrossSectionToData(cross_sections, data3d, data2d, lat, lon):
    
    valid_samps = np.all(cross_sections[0] > 100, axis=(0,1))
    cross_sections = cross_sections[:,:,:,valid_samps]
    ept = equivalentPotentialTempFromQ(t = cross_sections[0], q = cross_sections[1], PaPerLevel = cross_sections[-1]*100)

    # height, width, samples, variables
    ept = ept.reshape(*ept.shape, 1)
    mydata = ept
 
    all_seps = find3DSeparation(mydata, np.array([lat,lon]), debug = False)
    drawSeparationToData(all_seps, data3d, data2d,lat, lon, debug = False)

def create3DFrontPointClouds(cross_sections, lat, lon, level):
    tgt_array3d = np.zeros((1, len(level), len(lat), len(lon)))
    tgt_array2d = np.zeros((1, len(level), cross_sections.shape[2]))
    data_lon,data_lat = cross_sections[-2:].copy()
    if(cross_sections[:-2].size == 0):
        return tgt_array3d, tgt_array2d
    # we are assuming ERA5 data with a 0.25° resolution
    data_lon = ((data_lon+180)%360)-180
    # degree to pixel
    data_lat = (lat[0] - data_lat)*4  
    data_lon = (data_lon - lon[0])*4  
    # shape 0 consists of: num_level * num_atmospheric_variables + lat + lon 
    # => calculate num_vars
    num_vars = (cross_sections.shape[0]-2)//len(level)
    value_data = cross_sections[:-2].reshape(num_vars, -1, cross_sections.shape[-2], cross_sections.shape[-1])

    addAllCrossSectionToData(value_data, tgt_array3d, tgt_array2d, data_lat, data_lon)
    return tgt_array3d, tgt_array2d


def prepareData(infrontal_data, verbose):
    # adjust orientation
    # create new dictionary
    frontal_data = {}
    for key in infrontal_data:
        # fill with all keys
        frontal_data[key] = []
        for idx in range(len(infrontal_data[key])):
            if verbose:
                print("Preprocessing {} Front".format(key))

            # filter wrong results ( due to interpolation)
            data = infrontal_data[key][idx].copy()
            # add the data to each key
            frontal_data[key].append(data)
            orig = data.shape[-1]
            if orig > 0:
                # oversampling factor
                fac = 9 * 3

                # get normal direction
                if verbose:
                    print("mean pos {}/{}".format(data[-2].mean(), data[-1].mean()))
                
                    latlon = data[-2:,-1]-data[-2:,0]
                    latlon = latlon / np.linalg.norm(latlon, axis=0)
                    latlon = latlon.mean(axis = (-1))
                    print("Avg normal {}/{}".format(latlon[0], latlon[1]))

                data_fac = data.reshape(data.shape[0], data.shape[1], fac, -1)
                # in this case no sort happens
                data_fac_sort = data_fac

                # check neighboring directions
                directions = (data_fac_sort[-2:,-1]-data_fac_sort[-2:,0]).T
                directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
                directionsAng = np.angle(directions[:,fac//2,0]+1j*directions[:,fac//2,1])
                    
                # get local means
                stepsz = 20
                olddirs = np.zeros(2)
                valid_centers = ~np.any((data_fac_sort[0, :,fac//2] < 0), axis=0)
                # orient to local mean direction
                for i in range(0,directionsAng.shape[0],stepsz):
                    mdirsAng = (directionsAng[i:i+stepsz]+np.pi).mean(where= valid_centers[i:i+stepsz])-np.pi
                    mdirs = np.array([np.cos(mdirsAng),np.sin(mdirsAng)])
                    if i > 0:
                        if np.dot(mdirs, olddirs) < 0:
                            mdirs = -mdirs
                    olddirs = mdirs
                    dirs = np.dot(directions[i:i+stepsz],mdirs).T
                    flipsamples = dirs<0
                    subsamp = data_fac_sort[...,i:i+stepsz]
                    subsamp[:,:,flipsamples] = np.flip(subsamp[:,:,flipsamples], axis= 1)
                    data_fac_sort[...,i:i+stepsz] = subsamp
                
                data_sort = data_fac_sort.reshape(*data.shape[:-1],-1)
                valid_samps = ~np.any((data_sort[0] < 0), axis=(0))
                data_sort_valid= data_sort[...,valid_samps]
                if verbose:
                    print("filtered {} samples".format(((~valid_samps)*1).sum()))
                # set data
                frontal_data[key][idx] = data_sort_valid
    return frontal_data


def prepareAndConvertData(frontal_data, out, verbose = False):
    myfrontal_data = prepareData(frontal_data, verbose)
    convertData(myfrontal_data, out, verbose)


def frontsToNetCDF(out, tgt_lats, tgt_lons, tgt_levels, combineData, convertedFrontal_data3d = None):
    writer = netCDFWriter()
    axis_dict = {"time": [dt.datetime(2016, 1, 1,0)], "longitude": tgt_lons, "latitude": tgt_lats, "level": tgt_levels}
    variable_dict = {"front": combineData}
    unit_dict = {"latitude": "degrees_north", "longitude": "degrees_east", 
                "time": "hours since 1900-01-01", "level": "",
                "front": "1"}
    special_name_dict = {"latitude": "latitude", "longitude": "longitude", 
                "time": "time", "level": "model_level_number",
                "front": "front"}
    
    if not convertedFrontal_data3d is None:
        for i in convertedFrontal_data3d:
            variable_dict[i] = convertedFrontal_data3d[i]

            unit_dict[i] = "1"
            special_name_dict[i] = i+" front"
    
    writer.writeToFile(out, axis_dict, variable_dict, unit_dict, special_name_dict)


def convertData(myFrontal_data, out, verbose):
    levelAndVars,width,samples = myFrontal_data[list(myFrontal_data.keys())[0]][0].shape

    #### Cut of "uninteresting" part on the warm side of the front

    #### cold fronts are flipped before cutting the first half (+ a buffer)
    #### warm fronts are directly cut
    # overlap defines how much of the "uninteresing" part should be kept (some fronts do not behave as intended)
    Frontal_data = {}
    overlap = 40
    if "warm" in myFrontal_data:
        Frontal_data["warm"] = []
        #cut off the leading part of the warm front
        for idx in range(len(myFrontal_data["warm"])):
            Frontal_data["warm"].append(myFrontal_data["warm"][idx][:,myFrontal_data["warm"][idx].shape[1]//2-overlap:,:].copy())
    if "cold" in myFrontal_data:
        # cut off the trailing part of the cold front
        Frontal_data["cold"] = []
        for idx in range(len(myFrontal_data["cold"])):
            Frontal_data["cold"].append(np.flip(myFrontal_data["cold"][idx], axis=1)[:,myFrontal_data["cold"][idx].shape[1]//2-overlap:,:].copy())
    
    # area to display
    tgt_lats = np.arange(90,-0.1, -0.25)
    tgt_lons = np.arange(-80,20.1, 0.25)
    tgt_levels = np.array([500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000])

    # save counts in addition (to identify individual fronts later on)
    for x in Frontal_data:
        myd = []
        for smp in Frontal_data[x]:
            myd.append(smp.shape[-1])
        np.save(os.path.splitext(out)[0]+"_counts_{}.npy".format(x),np.array(myd),allow_pickle=False)
    # 3d Separation layer
    convertedFrontal_data3d = {x :np.zeros((1, tgt_levels.shape[0], tgt_lats.shape[0], tgt_lons.shape[0])) for x in Frontal_data}
    # 2d separation
    convertedFrontal_data2d = {x :np.zeros((1, tgt_levels.shape[0], 0)) for x in Frontal_data}


    for i in Frontal_data:
        for idx in range(len(Frontal_data[i])):
            if(Frontal_data[i][idx].shape[-1]>0):
                data3d, data2d = create3DFrontPointClouds(Frontal_data[i][idx], tgt_lats, tgt_lons, tgt_levels)
                convertedFrontal_data3d[i] += data3d[0:1]
                # data cut of "uninteresting part" needs to be corrected here
                data2d[0] += Frontal_data[i][idx].shape[1]-(2*overlap+1)
                # cold fronts were flipped before => flip coordinate
                if(i in ["cold"]):
                    data2d[0] = width-data2d[0]

                convertedFrontal_data2d[i] = np.concatenate((convertedFrontal_data2d[i], data2d[0:1]), axis=-1)
                
        # overlapping fronts should be corrected
        convertedFrontal_data3d[i] = (convertedFrontal_data3d[i] > 0) * 1
        if verbose:
            print("conversion {} done".format(i))

    mynewdata = (np.sum(np.array(list(convertedFrontal_data3d.values())), axis=0)>0)*1

    frontsToNetCDF(out, tgt_lats, tgt_lons, tgt_levels, mynewdata, convertedFrontal_data3d)
    for x in convertedFrontal_data2d:
        np.save(os.path.splitext(out)[0]+"_{}.npy".format(x),convertedFrontal_data2d[x],allow_pickle=False)
    print("writing file {} done".format(out))


