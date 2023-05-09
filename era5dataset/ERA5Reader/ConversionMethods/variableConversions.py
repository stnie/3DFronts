from metpy.calc import equivalent_potential_temperature, dewpoint_from_specific_humidity, dewpoint_from_relative_humidity, specific_humidity_from_dewpoint, relative_humidity_from_specific_humidity, relative_humidity_from_dewpoint
from metpy.units import units
import numpy as np


# Convert certain variables into others 

def dewp(mydat, writeVar, buffs, targetVars, *args):
    return dewpointTemp(mydat, False, buffs, targetVars)

def ept(mydat, writeVar, buffs, targetVars, *args):
    return equivalentPotentialTemp(mydat, False, buffs, targetVars)

def q(mydat, writeVar, buffs, targetVars, *args):
    return specificHumidity(mydat, False, buffs, targetVars)

def r(mydat, writeVar, buffs, targetVars, *args):
    return relativeHumidity(mydat, False, buffs, targetVars)




def dewPointTempFromQ(PaPerLevel,t,q):
    tu = units.Quantity(t, "K")
    qu = units.Quantity(q, "kg*kg**-1")
    pu = units.Quantity(PaPerLevel, "Pa")
    return dewpoint_from_specific_humidity(pu,tu,qu).magnitude

def dewPointTempFromR(t,r):
    tu = units.Quantity(t, "K")
    ru = units.Quantity(r, "%")
    return dewpoint_from_relative_humidity(tu,ru).magnitude


def equivalentPotentialTempFromQ(PaPerLevel, t, q):
    tu = units.Quantity(t, "K")
    qu = q
    pu = units.Quantity(PaPerLevel, "Pa")
    dewp = dewpoint_from_specific_humidity(pu,tu,qu)
    return equivalent_potential_temperature(pu,tu,dewp).magnitude
    
def equivalentPotentialTempFromR(PaPerLevel, t, r):
    tu = units.Quantity(t, "K")
    ru = units.Quantity(r, "%")
    pu = units.Quantity(PaPerLevel, "Pa")
    dewp = dewpoint_from_relative_humidity(tu, ru)
    return equivalent_potential_temperature(pu,tu,dewp).magnitude

def equivalentPotentialTempFromDewp(PaPerLevel, t, dewp):
    tu = units.Quantity(t, "K")
    dewp = units.Quantity(dewp, "degree_celsius")
    pu = units.Quantity(PaPerLevel, "Pa")
    return equivalent_potential_temperature(pu,tu,dewp).magnitude

def specificHumidityFromR(PaPerLevel, t, r):
    tu = units.Quantity(t, "K")
    ru = units.Quantity(r, "%")
    pu = units.Quantity(PaPerLevel, "Pa")
    dewp = dewpoint_from_relative_humidity(tu, ru)
    return specific_humidity_from_dewpoint(pu,dewp).magnitude

def specificHumidityFromDewp(PaPerLevel, dewp):
    dewp = units.Quantity(dewp, "degree_celsius")
    pu = units.Quantity(PaPerLevel, "Pa")
    return specific_humidity_from_dewpoint(pu,dewp).magnitude

def relativeHumidityFromQ(PaPerLevel, t, q):
    tu = units.Quantity(t, "K")
    qu = q
    pu = units.Quantity(PaPerLevel, "Pa")
    return relative_humidity_from_specific_humidity(pu,tu,qu).magnitude


# Used by the reader

def dewpointTemp(mydat, writeVar, buffs, targetVars):
    dewp = np.nan
    if("q" in targetVars or "var133" in targetVars):
        dewp = dewPointTempFromQ(buffs[0], buffs[1], buffs[2])
    elif("r" in targetVars):
        dewp = dewPointTempFromR(buffs[0], buffs[1])
    return val2Buf(mydat,dewp,writeVar)

def equivalentPotentialTemp(mydat, writeVar, buffs, targetVars):
    ept = np.nan
    if("q" in targetVars or "var133" in targetVars):
        ept = equivalentPotentialTempFromQ(buffs[0], buffs[1], buffs[2])
    elif("r" in targetVars or "RELHUM" in targetVars):
        ept = equivalentPotentialTempFromR(buffs[0], buffs[1], buffs[2])
    elif("dewp" in targetVars):
        ept = equivalentPotentialTempFromDewp(buffs[0], buffs[1], buffs[2])
    return val2Buf(mydat,ept,writeVar)

def specificHumidity(mydat, writeVar, buffs, targetVars):
    q = np.nan
    if("r" in targetVars or "RELHUM" in targetVars):
        q = specificHumidityFromR(buffs[0], buffs[1], buffs[2])
    elif("dewp" in targetVars):
        q = specificHumidityFromDewp(buffs[0], buffs[1])
    return val2Buf(mydat,q,writeVar)

def relativeHumidity(mydat, writeVar, buffs, targetVars):
    r = np.nan
    if("q" in targetVars or "var133" in targetVars):
        r = relativeHumidityFromQ(buffs[0], buffs[1], buffs[2])
    return val2Buf(mydat,r,writeVar)


def val2Buf(mydat,val,writeVar):
    if(writeVar):
        tgtBuffer = mydat
    else:
        tgtBuffer = np.zeros_like(mydat)
    tgtBuffer[:] = val
    return tgtBuffer






