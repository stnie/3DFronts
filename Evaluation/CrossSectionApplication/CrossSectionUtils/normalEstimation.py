from .samplingUtils import getKartesianCoordinatesFromSphere, rotateAroundX, rotateAroundY
import numpy as np

def getNormalEstimationAngular(pixY,pixX, offset, baseDeg, channelImage):
    negRang = 3
    posRang = negRang+1
    
    if(len(pixY)==0):
        return None, None, None, None

    myRegion = np.array([channelImage[pixY[y]-negRang:pixY[y]+posRang, pixX[y]-negRang:pixX[y]+posRang] for y in range(len(pixY))])
    # get all valid regions (at least 3 points)
    validRegs = np.sum(myRegion, axis=(1,2)) >= 3
    lostPoints = len(pixY)-np.sum(validRegs)
    # if no valid Region exists, we also return None
    if(lostPoints == len(pixY)):
        return None, None, None, None
    # filter the regions
    myRegion = myRegion[validRegs]
    # filter the points
    py = pixY[validRegs].astype(np.float32)
    px = pixX[validRegs].astype(np.float32)


    # for each py,px determine the normal of the tangential plane of the sphere
    # by getting the karthesian coordinates relative to the center
    lat = np.deg2rad(baseDeg[0]-((py+offset[0])*0.25).reshape(-1,1,1))
    # this is not as important, as only latitude generates a distortion
    lon = np.deg2rad(((px+offset[1])*0.25).reshape(-1,1,1))
    earth_radius = 6371

    # go through all points in the subregion
    sqrsize = (posRang+negRang)**2
    axsize = posRang+negRang
    yposs = ((np.arange(sqrsize)//(axsize)).reshape(1,myRegion.shape[1],myRegion.shape[2])-negRang)*0.25
    yposs *= -1
    xposs = ((np.arange(sqrsize)%(axsize)).reshape(1,myRegion.shape[1],myRegion.shape[2])-negRang)*0.25

    # xposs & yposs in rad
    xposs = np.deg2rad(xposs)+lon
    yposs = np.deg2rad(yposs)+lat

    samplePoints = getKartesianCoordinatesFromSphere(earth_radius, yposs, xposs)
    samplePoints = rotateAroundY(samplePoints, -lon)
    samplePoints = rotateAroundX(samplePoints, lat)

    
    # projection onto x-y plane == set z == radius
    samplePoints[2] = earth_radius
    xposs = samplePoints[0]    
    yposs = samplePoints[1]

    # oversampling (3 rotations, 9 translations = factor 27)
    pointsPerSide = 1
    repeatFactor1 = 2*pointsPerSide+1
    ydir,xdir = samplePointsToAngular(xposs,yposs,myRegion,negRang, repeatFactor1)
    
    repeatFactor = 9 
    py = np.tile(py,repeatFactor1*repeatFactor).reshape(repeatFactor,repeatFactor1,-1)
    px = np.tile(px,repeatFactor1*repeatFactor).reshape(repeatFactor,repeatFactor1,-1)
    ydir = np.tile(ydir,repeatFactor).reshape(repeatFactor1,repeatFactor,-1).transpose(1,0,2)
    xdir = np.tile(xdir,repeatFactor).reshape(repeatFactor1,repeatFactor,-1).transpose(1,0,2)
    stepDev = 1
    for o in range(repeatFactor):
        px[o] = (px[o]+((o-repeatFactor//2)*stepDev*ydir[o,repeatFactor1//2:repeatFactor1//2+1]))
        py[o] = (py[o]+((o-repeatFactor//2)*stepDev*xdir[o,repeatFactor1//2:repeatFactor1//2+1]))
    px = px.reshape(-1)
    py = py.reshape(-1)
    xdir = xdir.reshape(-1)
    ydir = ydir.reshape(-1)
    return ydir, xdir, py, px


def samplePointsToAngular(xposs,yposs, myRegion, negRang, repeatFactor = 1):
    angs = (np.rad2deg(np.angle(xposs+1j*yposs)))
    # center point should have no angle 
    myRegion[:,negRang,negRang] = 0
    # get the angle of all identified frontal points ( weighted by their strenght)
    filtAng = angs*myRegion
    # set center point to - 10000 (invalid angle) such that it is not used in the calculation)
    filtAng[myRegion==0] = -10000


    # determine number of samples in each quadrant 
    fstQ = np.sum((filtAng>0) * (filtAng<89), axis = (1,2)) 
    scdQ = np.sum((filtAng>91) * (filtAng<180), axis = (1,2)) 
    trdQ = np.sum(((filtAng>-180) * (filtAng<-91)), axis = (1,2)) 
    fthQ = np.sum((filtAng>-89) * (filtAng<0), axis = (1,2)) 
    fstT = fstQ
    scdT = scdQ
    fthT = fthQ
    trdT = trdQ
    # all sample points are within the upper half (first or second quadrant) (e.g. u/v shape with lowest point at 0,0)
    upper = (fstT>trdQ)*(fstT>fthQ)*(scdT>trdQ)*(scdT>fthQ)
    # points are mainly within the lower half (third or fourth quadrant) (e.q. n shape, with highest point at 0,0)
    lower = (fstQ<trdT)*(fstQ<fthT)*(scdQ<trdT)*(scdQ<fthT)
    rotSams = (upper | lower)
    
    # modulo 180 degree to get only orientation without direction
    # e.g. from south to north or from north to south become the same
    # also this removes the problem if angles are at -180 and 180 degree 
    angs = angs%180

    # mean direction of samples
    meanAng = np.mean((angs*myRegion), where = myRegion>0, axis=(1,2))
    
    # turn into radians
    # add 90 degree, as we are interested in the normal
    dirang = np.deg2rad(meanAng+90)
    # in the two special cases (all in upper or lower half) 
    # the modulo operator simply flipped the sample points. 
    # in this case the normal equals the mean + 180 degree
    dirang[rotSams] = np.deg2rad(meanAng[rotSams]+180)
    

    # oversampling
    multiDirang = np.tile(dirang,repeatFactor).reshape(repeatFactor, *dirang.shape)
    angleDev = np.deg2rad(30)
    for o in range(repeatFactor):
        multiDirang[o] += (o-repeatFactor/2)*angleDev
    dirang = multiDirang.reshape(repeatFactor*dirang.shape[0])
    myXdir, myYdir = np.cos(dirang), np.sin(dirang)
    
    # myYdir is oriented such that positive points towards north
    # Note that the image-coordinates are oriented sucht that positive points towards south
    return myYdir, myXdir
