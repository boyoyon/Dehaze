import cv2, glob, os, sys
import numpy as np

argv = sys.argv
argc = len(argv)

print('%s equaluzes histogram' % argv[0])
print('[usage] python %s <wildcard for images>' % argv[0])

if argc < 2:
    quit()

paths = glob.glob(argv[1])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))

for path in paths:

    base = os.path.basename(path)
    filename = os.path.splitext(base)[0]
    dst_path = 'clahe_%s.png' % filename

    src = cv2.imread(path)
    b, g, r = cv2.split(src)

    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)

    dst = cv2.merge((b, g, r))
    cv2.imwrite(dst_path, dst)
    print('save %s' % dst_path)




