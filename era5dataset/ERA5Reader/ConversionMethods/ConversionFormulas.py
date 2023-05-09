def setConversionFormulas():
    qFormulas = []
    eptFormulas = []
    dewpFormulas = []
    pressureFormulas = []
    rFormulas = []

    # build up list of formula & inherent calculations tuples
    # Assume pressure as "sp", lat and lon as "lat" and "lon"
    # If other names are used in the file, coordinate transforms are added to the formulas to automatically deduct this
    qFormulas += [("q(sp,t,r)",2), ("q(sp,dewp)",1),("Q",0), ("var133",0)]
    rFormulas += [("r(sp,t,q)",1),("RELHUM",0)]
    eptFormulas += [("ept(sp,t,r)",2), ("ept(sp,t,q)",2), ("ept(sp,t,dewp)",1)]
    dewpFormulas += [("dewp(sp,t,q)",1)]
    dewpFormulas += [("dewp(t,r)",1)]
    pressureFormulas += [("sp",0), ("plev",0)]
    
    # create a dictionary from those for each corresponding variable
    possForms = dict()

    # Calculations
    possForms["q"] = qFormulas
    possForms["ept"] = eptFormulas
    possForms["dewp"] = dewpFormulas
    possForms["r"] = rFormulas

    # Coordinate Axis transforms 
    possForms["sp"] = [("plev",0)]
    possForms["plev"] = [("sp",0)]
    possForms["lat"] = [("latitude",0)]
    possForms["lon"] = [("longitude",0)]
    possForms["latitude"] = [("lat",0)]
    possForms["longitude"] = [("lon",0)]

    # Variable renames
    possForms["t"] = [("T",0), ("var130",0)]
    possForms["z"] = [("FI",0), ("var129", 0)]
    possForms["u"] = [("U",0), ("var131", 0)]
    possForms["v"] = [("V",0), ("var132", 0)]
    possForms["var133"] = [("Q",0), ("q",0)]
    possForms["var130"] = [("T",0), ("t",0)]
    possForms["var129"] = [("FI",0), ("z", 0)]
    possForms["var131"] = [("U",0), ("u", 0)]
    possForms["var132"] = [("V",0), ("v", 0)]
    possForms["w"] = [("OMEGA",0)]
    return possForms
