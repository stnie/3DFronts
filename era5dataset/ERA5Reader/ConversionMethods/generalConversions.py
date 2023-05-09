import numpy as np

# General Conversions independent of what value is read


def pol2d(mydat, writeVar, buffs, targetVars, *args):
    return getPolar(buffs, mydat, True, False)
def abs2d(mydat, writeVar, buffs, targetVars, *args):
    return getPolar(buffs, mydat, False, False)
def delta_u(mydat, writeVar, buffs, targetVars, *args):
    return getDerivative(buffs, mydat, 2, False, None)
def delta_v(mydat, writeVar, buffs, targetVars, *args):
    return getDerivative(buffs, mydat, 1, False, None)


def getPolar(uvBuffer, mydat, angle, writeVar):
    uBuffer,vBuffer = uvBuffer

    if(writeVar):
        tgtBuffer = mydat
    else:
        tgtBuffer = np.zeros_like(mydat)
    # u and v buffer are filled
    if(angle):
        # The angle deviating counter clockwise from the east-west axis
        tgtBuffer[:] = np.abs(np.angle(uBuffer+1j*vBuffer))
    else:
        # The absolute signed by the north south direction
        tgtBuffer[:] = np.abs(uBuffer+1j*vBuffer)
    return tgtBuffer

def getDerivative(buffer, mydat, axis, writeVar, scaling ):
    
    # Post processing (e.g. derivative )
    if(writeVar):
        tgtBuffer = mydat
    else:
        tgtBuffer = np.zeros_like(mydat)
    if(axis == -2):
        tgtBuffer[:] = np.gradient(buffer[0], axis = 2)/scaling
    else:
        tgtBuffer[:] = np.gradient(buffer[0], axis= axis)
    return tgtBuffer