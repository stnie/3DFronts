
import numpy as np
# define sample positiosn and interpolation


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)
    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def getSamplePositionCircMulti(source, offset, ydir, xdir, num_points_per_side, distance, baseDeg = (90, -180), res=(-0.25, 0.25)):
    py = source[0]+offset[0]
    px = source[1]+offset[1]
    latbase,lonbase = baseDeg
    latres, lonres = res
    pyd = latbase + py*latres
    pxd = lonbase + px*lonres
    dist = (np.arange(-num_points_per_side, num_points_per_side+1,1)*distance).reshape(-1,1)
    dest = myOwnDestination(np.array([pyd,pxd]), dist, np.rad2deg(np.angle(ydir+1j*xdir)), True, False)
    xdists = (dest[:,:,1]-lonbase)/lonres-offset[1]
    ydists = (dest[:,:,0]-latbase)/latres-offset[0]
    return ydists, xdists

def rotateAroundZ(point, bearing):
    newX = np.cos(bearing)*point[0] - np.sin(bearing)*point[1]
    newY = np.sin(bearing)*point[0]+np.cos(bearing)*point[1]
    newZ = point[2]
    return np.array([newX,newY,newZ])

def rotateAroundY(point, bearing, broadCastToShape = False):
    newX = np.cos(bearing)*point[0] + np.sin(bearing)*point[2]
    if(broadCastToShape):
        newY = np.broadcast_to(point[1], bearing.shape)
    else:
        newY = point[1]
    newZ = -np.sin(bearing)*point[0] + np.cos(bearing)*point[2]
    return np.array([newX,newY,newZ])

def rotateAroundX(point, bearing):
    newX = point[0]
    newY = -np.sin(bearing)*point[2]+np.cos(bearing)*point[1]
    newZ = np.cos(bearing)*point[2] + np.sin(bearing)*point[1]
    return np.array([newX,newY,newZ])

    
def getEllipseRadiusAtPoint(rmaj, rmin, theta, phi):
    # theta is elevation (-pi/2 , pi/2)
    # phi is azimuth, right (-pi, pi)
    a = rmaj
    b = rmin
    ttan = np.tan(theta)
    z = 1/(a*a) + (ttan*ttan)/(b*b)
    x = np.sqrt(1/z)
    y = x*ttan
    r = np.sqrt(x*x+y*y)
    return r


def getKartesianCoordinatesFromSphere(r, theta, phi):
    # theta is elevation (-pi/2 , pi/2)
    # phi is azimuth, right (-pi, pi)
    z = r*np.cos(phi)*np.cos(theta)
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(theta)
    # x y z correspond to right, up, front (right hand rule)
    return np.array([x,y,z])

def getSphereCoordinatesFromKartesian(point):
    r = np.linalg.norm(point, axis = 0)
    theta = np.arcsin(point[1]/r)
    phi = np.arctan2(point[0], point[2])
    return np.array([r,theta, phi])


def myOwnDestination(latlonstart, distance, bearing, geodesic = True, flat_east = False):
    rlatlonstart = np.deg2rad(latlonstart)
    rbearing = np.deg2rad(90-bearing)
    if(geodesic):
        earth_semi_major = 6378
        earth_semi_minor = 6357
        # get local radius of the ellipse (estimate)
        radius = getEllipseRadiusAtPoint(earth_semi_major, earth_semi_minor,*rlatlonstart)
    else:
        # take average earth radius
        radius = 6371
    
    # sample points where east is 90° rotation from north on the sphere (e.g. at the poles walking east is equal to walking straight towards the equator)
    return getDestinationSpherical(rlatlonstart, rbearing, distance, radius)

def getDestinationSpherical(latlonstart, bearing, distance, radius):
    kartLatLonStart = np.zeros((3, latlonstart.shape[-1]))
    kartLatLonStart[2] = radius
    # get angle to rotate on the plane (locally)
    angle = distance / radius

    # rotate start position by angle along rotation plane 
    rotKartEnd = rotateAroundY(kartLatLonStart, angle, True)
    
    # rotate the "plane" according to bearing
    kartEnd = rotateAroundZ(rotKartEnd, bearing)
    # rotate resulting vector onto the original coordinate system, where latlonstart is located at lat , lon
    kartEnd = rotateAroundX(kartEnd, -latlonstart[0])
    kartEnd = rotateAroundY(kartEnd, latlonstart[1])

    # get the sphere coordinates (radius, lat, lon) to return the target point
    latlonEnd = np.rad2deg(getSphereCoordinatesFromKartesian(kartEnd))
    return np.array([latlonEnd[1], latlonEnd[2]]).T

