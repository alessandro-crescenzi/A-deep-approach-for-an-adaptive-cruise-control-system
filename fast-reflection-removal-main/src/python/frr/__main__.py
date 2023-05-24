import torch
import time
from frr import FastReflectionRemoval
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def main():
    # read image and normalize it into [0, 1]
    img = plt.imread('0.png')

    M, N, _ = img.shape

    frr = FastReflectionRemoval(h=0.03, M=M, N=N)

    img = torch.from_numpy(img).to("cuda:0")

    start = time.time()
    # remove reflection
    result_img = frr.remove_reflection(img)
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))

    plt.imsave("out.png", result_img.numpy())


if __name__ == "__main__":
    main()
