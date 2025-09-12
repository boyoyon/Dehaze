import cv2, os, sys
import numpy as np

def Complex2MagPhase(complex_image):

    #complex_image = np.fft.fftshift(complex_image)
    mag_image = cv2.magnitude(complex_image[:,:,0], complex_image[:,:,1])
    phase_image = np.angle(complex_image[:,:,0] + 1j * complex_image[:,:,1])

    return mag_image, phase_image

def MagPhase2Complex(mag_image, phase_image):

    real_part = mag_image * np.cos(phase_image)
    imaginary_part = mag_image * np.sin(phase_image)

    complex_image = cv2.merge((real_part, imaginary_part))
    #complex_image = np.fft.fftshift(complex_image)

    return complex_image 

def main():

    start = 0.1
    end = 1.0
    step = 0.1

    argv = sys.argv
    argc = len(argv)

    if argc < 2:
        print('%s enhases magnitude of the image' % argv[0])
        print('%s <image> [<start> <end> <step>]' % argv[0])
        quit()

    image_path = argv[1]
    base = os.path.basename(image_path)
    filename, _ = os.path.splitext(base)

    if argc > 2:
        start = float(argv[2])

    if argc > 3:
        end = float(argv[3])

    if argc >4:
        step = float(argv[4])

    print('start:%f, end:%f, step:%f' % (start, end, step))

    src = cv2.imread(image_path)
    src = src.astype(np.float32) / 255.0
    
    b, g, r = cv2.split(src)

    dftB = cv2.dft(b, flags = cv2.DFT_COMPLEX_OUTPUT)
    magB, phaseB = Complex2MagPhase(dftB)

    dftG = cv2.dft(g, flags = cv2.DFT_COMPLEX_OUTPUT)
    magG, phaseG = Complex2MagPhase(dftG)
    
    dftR = cv2.dft(r, flags = cv2.DFT_COMPLEX_OUTPUT)
    magR, phaseR = Complex2MagPhase(dftR)
    
    images = []
    for MagScale in np.arange(start, end + step, step): 

        compB = MagPhase2Complex(magB * MagScale, phaseB)
        idftB = cv2.idft(compB)
        B = cv2.magnitude(idftB[:,:,0], idftB[:,:,1])
        B /= np.max(magB)
    
        compG = MagPhase2Complex(magG * MagScale, phaseG)
        idftG = cv2.idft(compG)
        G = cv2.magnitude(idftG[:,:,0], idftG[:,:,1])
        G /= np.max(magG)
    
        compR = MagPhase2Complex(magR * MagScale, phaseR)
        idftR = cv2.idft(compR)
        R = cv2.magnitude(idftR[:,:,0], idftR[:,:,1])
        R /= np.max(magR)
    
        image = cv2.merge((B, G, R))
        image *= 255
        image[image > 255] = 255
        image = image.astype(np.uint8)
        images.append(image)

    mergeMertens = cv2.createMergeMertens()
    fusion = mergeMertens.process(images)
    cv2.imwrite('MagScaleFusion.png', fusion * 255)
    print('save MagScaleFusion.png')
    cv2.imshow('fusion', fusion)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
