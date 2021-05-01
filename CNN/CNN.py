from utilities.std_imports import *
import sys

def conv(img, filter):
    filterSize = filter.shape[1]
    featMap = numpy.zeros((img.shape))
    # Convolution operation in the image
    for r in numpy.uint16(numpy.arange(filterSize/2.0, img.shape[0] - filterSize/2.0 + 1)):
        for c in numpy.uint16(numpy.arange(filterSize/2.0, img.shape[1] - filterSize/2.0 + 1)):
            # Getting the current region to apply the filter
            floor = numpy.uint16(numpy.floor(filterSize/2.0))
            ceil = numpy.uint16(numpy.ceil(filterSize/2.0))
            currRegion = img[r-floor:r+ceil , c-floor:c+ceil]
            
            # Element-wise product 
            currRes = currRegion * filter
            convSum = numpy.sum(currRes)
            featMap[r, c] = convSum 
            
    # Clipping the outliers
    halfFilterSize = numpy.uint16(filterSize/2.0)
    return featMap[halfFilterSize:featMap.shape[0]-halfFilterSize , halfFilterSize:featMap.shape[1]-halfFilterSize]

# Convolution operation
def Conv(img, filter):
    # Checks: 1.  number of image channels matches the filter depth 2. filter dimensions are equal and 3. are odd
    if (len(img.shape) > 2 or len(filter.shape) > 3) and img.shape[-1] != filter.shape[-1]: print("Error: Number of channels in both image and filter must match."); sys.exit()
    if filter.shape[1] != filter.shape[2]: print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.'); sys.exit()
    if filter.shape[1]%2==0: print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.'); sys.exit()

    featMaps = numpy.zeros((img.shape[0]-filter.shape[1]+1 , img.shape[1]-filter.shape[1]+1 , filter.shape[0]))

    # Convolving
    for f in range(filter.shape[0]):
        print("Filter ", f+1)
        currFilter = filter[f, :] 
        
        # Checking if there are mutliple channels for the single filter (to convolve in each channel)
        if len(currFilter.shape) > 2:
            convMap = conv(img[:, :, 0], currFilter[:, :, 0]) # Array holding the sum of all feature maps.
            for ch in range(1, currFilter.shape[-1]): 
                convMap = convMap + conv(img[:, :, ch], currFilter[:, :, ch])
        else: 
            convMap = conv(img, currFilter)
        featMaps[:, :, f] = convMap 
    return featMaps 
    

# Pooling operation
def Pooling(featMap, size=2, stride=2):
    poolOut = numpy.zeros((numpy.uint16((featMap.shape[0]-size+1)/stride+1) , numpy.uint16((featMap.shape[1]-size+1)/stride+1), featMap.shape[-1]))
    for map_num in range(featMap.shape[-1]):
        r2 = 0
        for r in numpy.arange(0,featMap.shape[0]-size+1, stride):
            c2 = 0
            for c in numpy.arange(0, featMap.shape[1]-size+1, stride):
                poolOut[r2, c2, map_num] = numpy.max([featMap[r:r+size,  c:c+size]])
                c2 += 1
            r2 += 1
    return poolOut

# Relu activation function
def Relu(featMap):
    reluOut = numpy.zeros(featMap.shape)
    maps = featMap.shape[-1]
    for m in range(maps):
        for r in numpy.arange(0,featMap.shape[0]):
            for c in numpy.arange(0, featMap.shape[1]):
                reluOut[r, c, m] = numpy.max([featMap[r, c, m], 0])
    return reluOut
