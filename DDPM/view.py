import numpy
import cv2

if __name__ == '__main__':
    samples = numpy.load("./unconditional_cifar10_samples.npy")
    for s in samples:
        image = (s + 1.0) / 2
        image = cv2.resize(image, (500, 500))
        cv2.imshow("Display", image)
        cv2.waitKey()