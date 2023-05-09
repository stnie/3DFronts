
import netCDF4 as nc4
import numpy as np

class netCDFWriter:
    def __init__(self, format="NETCDF4_CLASSIC", infinite_axis = "time", title = "custom_file"):
        self.format = format
        self.infinite_axis = infinite_axis
        self.title = title
    
    def writeToFile(self, filename, axis_dict, variable_dict, unit_dict, special_name_dict = None):
        ncfile = nc4.Dataset(filename, mode="w", format=self.format)

        # create dims
        axes = {}
        for axis_name,axis_value in axis_dict.items():
            if(axis_name == self.infinite_axis):
                axes[axis_name] = ncfile.createDimension(axis_name, None)
            else:
                axes[axis_name] = ncfile.createDimension(axis_name, axis_value.shape[0])

        #create title
        ncfile.title = self.title

        #create variables
        # Axis Variables
        ax_vars = {}
        for axis_name,axis_value in axis_dict.items():
            ax_vars[axis_name] = ncfile.createVariable(axis_name, np.float32, (axis_name,))
            if unit_dict is None or not axis_name in unit_dict:
                # use some defaults
                if(axis_name in ["lat", "latitude"]):
                    ax_vars[axis_name].units = "degrees_north"        
                elif(axis_name in ["lon", "longitude"]):
                    ax_vars[axis_name].units = "degrees_east"        
                elif(axis_name in ["level"]):
                    ax_vars[axis_name].units = "level"        
                elif(axis_name in ["plev"]):
                    ax_vars[axis_name].units = "plev"        
                elif(axis_name in ["time"]):
                    ax_vars[axis_name].units = "hours since 1900-01-01 00:00:00.0"        
                else:
                    print("cannot find default for axis {}. End writing!".format(axis_name))
                    ncfile.close()
                    return
                print("Found default unit {} for axis {}".format(ax_vars[axis_name].units, axis_name))
            else:
                ax_vars[axis_name].units = unit_dict[axis_name]
            if not special_name_dict is None and axis_name in special_name_dict:
                ax_vars[axis_name].long_name = special_name_dict[axis_name]
                #ax_vars[axis_name].standard_name = special_name_dict[axis_name]
            
            # because time is annoying...
            if(axis_name == "time"):
                ax_vars[axis_name][:] = nc4.date2num(axis_value, ax_vars[axis_name].units)
            else:
                ax_vars[axis_name][:] = axis_value

        fvars = {}
        for variable_name,variable_value in variable_dict.items():
            fvars[variable_name] = ncfile.createVariable(variable_name, np.float32, ("time","level","latitude","longitude"))
            if not variable_name in unit_dict:
                fvars[variable_name].units = ""
            else:
                fvars[variable_name].units = unit_dict[variable_name]
            if not special_name_dict is None:
                if variable_name in special_name_dict:
                    fvars[variable_name].standard_name = special_name_dict[variable_name]
            fvars[variable_name][:] = variable_value

        ncfile.close()
