from .L137Levels import L137Calculator
import numpy as np

#####
# Convert coordinates into others
#####

def LatTokmPerLon(lat_data, lonResolution = 0.25):
    unit = 1000
    radius = 6365831.0
    r = (radius*np.cos(lat_data/180*np.pi))/unit
    lat_data = r*2*np.pi*lonResolution / 360
    return lat_data

def calculatePressureFromML(data, levels):
    lc = L137Calculator()
    lc.getPressureAtMultipleLevels(data[-1], levels, data)

