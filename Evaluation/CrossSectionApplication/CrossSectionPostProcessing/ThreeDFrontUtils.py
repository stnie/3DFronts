import numpy as np
from scipy.ndimage import convolve1d

def varianceError(image):
    maxScore = 10000
    sample_width = 20
    leftVars = np.ones_like(image)*maxScore
    rightVars = np.ones_like(image)*maxScore
    leftMeans = np.ones_like(image)*maxScore
    rightMeans = np.ones_like(image)*maxScore
    mask = np.ones(sample_width)/sample_width
    
    paddimg = np.concatenate((image, np.zeros((image.shape[0],sample_width,*image.shape[2:]))), axis=1)
    cmeanleft = convolve1d(paddimg, mask,origin= -(mask.shape[0]//2), axis=1, mode = 'constant')
    sumsleft = convolve1d(paddimg**2, mask,origin= -(mask.shape[0]//2), axis=1, mode = 'constant')
    leftDivs = np.arange(1,sample_width).reshape(1,-1,1,1)
    rightDivs = np.flip(leftDivs, axis=1)
    cmeanleft[:,:sample_width-1] *= sample_width/leftDivs
    cmeanleft[:,-sample_width+1:] *= sample_width/rightDivs
    sumsleft[:,:sample_width-1] *= sample_width/leftDivs
    sumsleft[:,-sample_width+1:] *= sample_width/rightDivs
    
    cvarleft = sumsleft-cmeanleft*cmeanleft
    leftVars[:,1:] = cvarleft[:,:-sample_width-1]
    leftMeans[:,1:] = cmeanleft[:,:-sample_width-1]
    rightVars[:,:-1] = cvarleft[:, sample_width:-1]
    rightMeans[:,:-1] = cmeanleft[:,sample_width:-1]
    
    var_scores = np.mean(np.linalg.norm([leftVars, rightVars], axis= (0)), axis=-1)
    grads = gradientError(image)
    # all samples are oriented in such a way that left side is supposed to be the warm side 
    # and the right side is supposed to be the cold side of the front.
    # this term further encourages the network to prefer such conditions 
    frontCharacteristic = (rightMeans - leftMeans).mean(axis=-1)

    
    scores = np.zeros_like(var_scores)
    
    var_weight, grad_weight, fchar_weight = 1, 1, 1
    scores += var_scores * var_weight
    scores += grads * grad_weight
    scores += frontCharacteristic * fchar_weight

    # use gradient 10% quantile criterion as a filter (only the 10% steepest negative gradients are considered valid front positions)
    gradFilterValue = np.quantile(grads,0.1, axis=(1,2), keepdims=True)
    scores[grads>=gradFilterValue] = maxScore
    scores[:,0] = maxScore
    scores[:,-1] = maxScore

    return scores

def gradientError(image):
    # minimize absolute gradient error
    return np.mean(np.gradient(image, axis=1), axis=-1)


def CreateScoringMatrix(in_image, coords, debug = False):
    image = np.flip(in_image, axis=0)
    scores = varianceError(image)
    # scoring matrix created
    # add some regularizations for variance
    return evaluateScoringMatrix(image, coords, scores, debug)


def evaluateScoringMatrix(image, coords, scores, debug = False):
    height, width, samples, channels = image.shape
    maxAllowedScore = 1000
    # height before sample region is adjusted
    height_step = 1
    offset = 20
    # start position of a sample region
    oldLeft = np.ones((samples), dtype=np.int32)*offset
    # width of are whithin the optimal score is searched
    score_width = 40

    bestidx = np.ones((image.shape[0],image.shape[2]))*np.nan

    local_scores = np.zeros((height_step, score_width, samples))

    # try to find optimal points, locally restricted

    local_image = image.copy()[:,:,samples//2,0]
    local_scorings = scores.copy()[:,:,samples//2]
    local_masks = np.zeros((height, width))
    local_results = np.zeros((height,width))
    for h in range(0,image.shape[0],height_step):
        level_start = h
        level_end = min(image.shape[0], h+height_step)
        current_height = level_end-level_start
        leftEnd = oldLeft
        rightEnd = leftEnd + score_width

        # build local scores:
        for s in range(samples):
            local_scores[:current_height,:,s] = scores[level_start:level_end, leftEnd[s]:rightEnd[s],s]
            
        # get best Idx for pth place
        indices = np.argmin(local_scores[:current_height], axis = 1)
        best_scores = np.min(local_scores[:current_height], axis = 1)
        if debug:
            if h == 0: 
                debug_pos = (indices == score_width-1) + (indices == score_width-2)
                if np.any(debug_pos):
                    positions = debug_pos[0]
                    print(coords.shape, positions.shape)
                    print(coords[:, offset+indices[0,positions], positions])
            local_masks[h, leftEnd[samples//2]:rightEnd[samples//2]] = 1

        # only evaluate scores that fulfill the maxAllowedScore threshold (throw out invalid locations)
        goodScores = best_scores <= maxAllowedScore
        bestidx[level_start:level_end][goodScores] = indices[goodScores]

        
        pnts = np.mean(bestidx[level_start:level_end], axis=(0))
        leftOff = np.clip(np.nan_to_num(pnts, nan= 0).astype(np.int32)-score_width//2+0, -score_width//2, score_width//2)
        if(np.any(np.isnan(pnts))):
            # for all samples
            for p in range(pnts.shape[0]):
                if(np.isnan(pnts[p])):
                    leftOff[p] = 0
        # adjust sample image position based on the placement of the separation layers
        bestidx[level_start:level_end] += oldLeft[None,:]
        oldLeft = np.clip(oldLeft+leftOff,0, image.shape[1]-score_width-1)
    
    if(debug):
        for h in range(height):
            myPos = bestidx[h,samples//2]
            if not np.isnan(myPos):
                myPosInt = myPos.astype(np.int32)
                local_results[h, myPosInt] = 1

        np.save("debug_local_image.npy", local_image)
        np.save("debug_local_scores.npy", local_scorings)
        np.save("debug_local_masks.npy", local_masks)
        np.save("debug_local_results.npy", local_results)
    # readjust orientation (top to bottom)
    bestidx = np.flip(bestidx, axis = 0)
    return bestidx






def find3DSeparation(images, coords, debug = False):
    height, width, samples, variables = images.shape
    bestidx = np.ones((height, samples), dtype=np.int32)*-1
    
    # initial separation for each individually
    bestidx = CreateScoringMatrix(images, coords, debug = debug)
    
    if debug:
        bestidx[:] = 40
    return bestidx

