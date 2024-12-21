import cv2, os, sys
import numpy as np
from cv2.ximgproc import guidedFilter

WINDOW_SIZE = 45
WEIGHT = 70
SHRINK_SCALE = 5

def create_min_image(src, window_size):

    half = window_size // 2

    height, width = src.shape[:2]

    dst = np.empty(src.shape, src.dtype)
    b, g, r = cv2.split(src)

    for y in range(height):
        startY = np.max((0, y - half))
        endY = np.min((height, y + half))

        for x in range(width):
            startX = np.max((0, x - half))
            endX = np.min((width, x + half))

            patch = b[startY:endY, startX:endX]
            dst[y][x][0] = np.min(patch)

            patch = g[startY:endY, startX:endX]
            dst[y][x][1] = np.min(patch)

            patch = r[startY:endY, startX:endX]
            dst[y][x][2] = np.min(patch)

    return dst

def create_max_image(src, window_size):

    half = window_size // 2

    height, width = src.shape[:2]

    dst = np.empty(src.shape, src.dtype)
    b, g, r = cv2.split(src)

    for y in range(height):
        startY = np.max((0, y - half))
        endY = np.min((height, y + half))

        for x in range(width):
            startX = np.max((0, x - half))
            endX = np.min((width, x + half))

            patch = b[startY:endY, startX:endX]
            dst[y][x][0] = np.max(patch)

            patch = g[startY:endY, startX:endX]
            dst[y][x][1] = np.max(patch)

            patch = r[startY:endY, startX:endX]
            dst[y][x][2] = np.max(patch)

    return dst

def getAirLight(darkchannel):

    height, width = darkchannel.shape[:2]

    histogram = np.zeros((256,3), np.int32)

    for y in range(height):
        for x in range(width):
            b = darkchannel[y][x][0]
            g = darkchannel[y][x][1]
            r = darkchannel[y][x][2]

            histogram[b][0] += 1
            histogram[g][1] += 1
            histogram[r][2] += 1

    th = width * height // 1000

    boolB = boolG = boolR = False
    numB = numG = numR = 0
    sumB = sumG = sumR = 0

    for i in reversed(range(256)):
        if not boolB:
            numB += histogram[i][0]        
            sumB += histogram[i][0] * i
            if numB > th:
                boolB = True

        if not boolG:
            numG += histogram[i][1]
            sumG += histogram[i][1] * i
            if numG > th:
                boolG = True

        if not boolR:
            numR += histogram[i][2]
            sumR += histogram[i][2] * i
            if numR > th:
                boolR = True

        if boolB and boolG and boolR:
            break

    return (sumB//numB, sumG//numG, sumR//numR)

def getTmax(darkchannel, airlight, weight):

    height, width = darkchannel.shape[:2]

    tmax = np.empty((height, width, 3), np.uint8)

    for y in range(height):
        for x in range(width):
            tmax[y][x][0] = (255 * 100 - weight * 255 * darkchannel[y][x][0] // airlight[0]) // 100
            tmax[y][x][1] = (255 * 100 - weight * 255 * darkchannel[y][x][1] // airlight[1]) // 100
            tmax[y][x][2] = (255 * 100 - weight * 255 * darkchannel[y][x][2] // airlight[2]) // 100

    return tmax

def dehaze(src, tmax, airlight):

    if src.shape[0] != tmax.shape[0] or src.shape[1] != tmax.shape[1]:
        print('dimension mismatch')
        return None

    height, width = src.shape[:2]

    dst = np.empty((height, width, 3), np.uint8)
 
    for y in range(height):
        for x in range(width):
            SRCb = src[y][x][0]
            SRCg = src[y][x][1]
            SRCr = src[y][x][2]

            DSTb = ((SRCb - airlight[0]) * 255 + tmax[y][x][0] * airlight[0]) // tmax[y][x][0]
            DSTg = ((SRCg - airlight[1]) * 255 + tmax[y][x][1] * airlight[1]) // tmax[y][x][1]
            DSTr = ((SRCr - airlight[2]) * 255 + tmax[y][x][2] * airlight[2]) // tmax[y][x][2]

            dst[y][x][0] = min(255, max(0, DSTb))
            dst[y][x][1] = min(255, max(0, DSTg))
            dst[y][x][2] = min(255, max(0, DSTr))

    return dst

def main():

    global WEIGHT, WINDOW_SIZE

    argv = sys.argv
    argc = len(argv)
    
    if argc < 2:
        print('%s dehazes hazy image' % argv[0])
        print('%s <image> [<weight>] [<window_size>]' % argv[0])
        quit()
    
    src = cv2.imread(argv[1])
    
    height_src, width_src = src.shape[:2]
    
    if argc > 2:
        WEIGHT = int(argv[2])
    
    if argc > 3:
        WINDOW_SIZE = int(argv[3])
    
    if WINDOW_SIZE < 10:
        WINDOW_SIZE = 10
    
    freq = cv2.getTickFrequency()
    count0 = cv2.getTickCount()
    
    height_shrink = height_src // SHRINK_SCALE
    width_shrink = width_src // SHRINK_SCALE
    
    shrinked = cv2.resize(src, (width_shrink, height_shrink))
    
    min_image = create_min_image(shrinked, WINDOW_SIZE // SHRINK_SCALE)
    max_image = create_max_image(min_image, WINDOW_SIZE // SHRINK_SCALE)
    
    expanded = cv2.resize(max_image, (width_src, height_src))
    
    print('(1/5) Darkchannel created')
    
    size = 60
    sigma = 1e-6 * 255 * 255
    darkchannel = cv2.ximgproc.guidedFilter(src, expanded, size, sigma)
    
    print('(2/5) Darkchannel refined')
    
    AirLight = getAirLight(darkchannel)
    
    print('(3/5) Air Light calculated')
    
    tmax = getTmax(darkchannel, AirLight, WEIGHT)
    
    print('(4/5) tmax created')
    
    dehazed = dehaze(src, tmax, AirLight)
    
    print('(5/5) dehaze completed')
    print('Hit any key to terminate this program')
    
    count1 = cv2.getTickCount()
    print((count1 - count0) / freq)
    
    cv2.imshow('source', src)
    cv2.imshow('dehazed', dehazed)
    
    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = 'dehazed_%s.png' % filename
    cv2.imwrite(dst_path, dehazed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()

