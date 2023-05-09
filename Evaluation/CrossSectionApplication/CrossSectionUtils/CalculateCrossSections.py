import numpy as np
from .samplingUtils import *
from .normalEstimation import getNormalEstimationAngular


# calculate cross sections for a given image 


def getHorizontalSampleLocations(channelImage,blength, udir, vdir, num_points_per_side, distance_between_points, offset, baseDeg = (90,-180), res = (-0.25,0.25),  getDirs = False, spherical = False, random_points = True, rng = None):
    #####
    # image space:
    # y corresponds to latitudinal direction (e.g. north to south) ~ -vdir of wind
    # x corresponds to longitudinal direction ( e.g. west to east) ~ udir of wind
    # py, px, pointsY and pointsX are considered to be oriented according to image space

    # world space:
    # y corresponds to latitudinal direction (e.g. south to north) ~ vdir of wind
    # x corresponds to longitudinal direction ( e.g. west to east) ~ udir of wind
    # udir,vdir, myXdir, myYdir are considered to be oriented according to world space
    # (i.e. y-component is flipped relative to image space)
    wind = np.array([udir,vdir])
    udir, vdir = wind / (np.linalg.norm(wind, axis=0)+0.0000001)
    validPoints = np.nonzero(channelImage[blength:-blength,blength:-blength])

    localNumPoints = np.minimum(validPoints[0].shape[0], 300000)
    pixY, pixX = blength+validPoints[0][:localNumPoints], blength+validPoints[1][:localNumPoints]
    # estimate the normal direction and filter all points where a normal cannot be estimated (due to insufficient points in neighborhood)
    # myYdir is oriented such that north is positive, and south is negative!
    
    if(random_points):
        if(localNumPoints < 100):
            print("some samples where invalid. Only found {} valid points".format(localNumPoints))
        randomAngle = rng.random(localNumPoints)*2*np.pi
        myYdir, myXdir = np.sin(randomAngle), np.cos(randomAngle)
        py, px = pixY.astype(np.float32), pixX.astype(np.float32)
    else:
        myYdir,myXdir, py, px = getNormalEstimationAngular(pixY,pixX,offset, baseDeg, channelImage)

    if(myXdir is None or myYdir is None):
        #ignore the point and continue with the next
        print("Frontal type does not exist")
        if(getDirs):
            return None, None, None, None
        else:
            return None, None
    # transform py,px coordinates to world space
    #  sample coordinates along myXYdir
    # transform sampled coordinates back to image space
    pointsY, pointsX = getSamplePositionCircMulti((py,px), offset, myYdir, myXdir, num_points_per_side, distance_between_points, baseDeg, res)
    
    usamp = bilinear_interpolate(udir, pointsX, pointsY)
    vsamp = bilinear_interpolate(vdir, pointsX, pointsY)
    epsilon = 1e-8
    wlen = np.sqrt(usamp*usamp+vsamp*vsamp)+epsilon
    usamp /= wlen
    vsamp /= wlen
    
    
    xdiffs = (pointsX-px.reshape(-1,1))
    xdiffs[:,:num_points_per_side] *= -1
    xdiffs[:,num_points_per_side] = np.mean(xdiffs, axis=1)

    ydiffs = (pointsY-py.reshape(-1,1))
    ydiffs[:,:num_points_per_side] *= -1
    ydiffs[:,num_points_per_side] = np.mean(ydiffs, axis=1)
    ydiffs *= -1
        
    difflens = np.sqrt(xdiffs*xdiffs+ydiffs*ydiffs)
    valdiffs = difflens != 0
    xdiffs[valdiffs] /= difflens[valdiffs]
    ydiffs[valdiffs] /= difflens[valdiffs]
    # get the dot product of mean wind along the normal and the normal
    # to determine whether or not both are in the same direction
    direction = np.mean(xdiffs*usamp + ydiffs*vsamp, axis = 1)

    # If the dot product is negative, the sampling points are oriented wrongly 
    # and need to be flipped
    negDir = direction <= 0
    pointsX[negDir] = np.flip(pointsX[negDir], axis=1)
    pointsY[negDir] = np.flip(pointsY[negDir], axis=1)

    # only consider points where the angle between the wind and the normal is
    # not (near) orthogonal. As in those cases the direction based sorting 
    # is not very reliable, as the wind is not a clear indicator of direction
    # currently not used
    goodDir = np.abs(direction) >= -1#np.cos(80/180*np.pi)
    pointsX = pointsX[goodDir]
    pointsY = pointsY[goodDir]

    if getDirs:
        return pointsX, pointsY, np.abs(direction[goodDir]), np.rad2deg(np.angle(myYdir[goodDir]+1j*myXdir[goodDir]))
    else:
        return pointsX, pointsY


def getValAlongNormalWithDirs(image, var, udir, vdir, num_points_per_side, distance_between_points, border, offset, baseDeg = (90,-180), res = (-0.25,0.25), random_points=False, rng = None):
    blength = max(border,num_points_per_side)
    pointsX, pointsY, windToNormal, normalDir = getHorizontalSampleLocations(image,blength, udir, vdir, num_points_per_side, distance_between_points, offset, baseDeg, res, True, True, random_points, rng)
    if(pointsX is None):
        numPoints = 0
        avgVar = np.zeros((var.shape[0],2*num_points_per_side+1, numPoints))
        windToNormal = np.zeros(numPoints)
        normalDir = np.zeros(numPoints)
    else:
        withinArea = np.all((pointsX<(udir.shape[1]-1)) * (pointsX >= 0) * (pointsY < (udir.shape[0]-1)) * (pointsY >= 0), axis= 1)
        numPoints = pointsX.shape[0] 
        avgVar = np.zeros((var.shape[0],2*num_points_per_side+1, numPoints))
        for i in range(avgVar.shape[0]):
            avgVar[i] = (bilinear_interpolate(var[i], pointsX, pointsY)).T
            avgVar[i,...,~withinArea] = -20000
    return avgVar, windToNormal, normalDir, numPoints

