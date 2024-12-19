import cv2, glob, os, sys
import numpy as np

argv = sys.argv
argc = len(argv)

print('%s equaluzes histogram' % argv[0])
print('[usage] python %s <wildcard for images>' % argv[0])

if argc < 2:
    quit()

paths = glob.glob(argv[1])

for path in paths:

    base = os.path.basename(path)
    filename = os.path.splitext(base)[0]
    dst_path = 'he_%s.png' % filename

    src = cv2.imread(path)
    b, g, r = cv2.split(src)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    dst = cv2.merge((b, g, r))
    cv2.imwrite(dst_path, dst)
    print('save %s' % dst_path)




