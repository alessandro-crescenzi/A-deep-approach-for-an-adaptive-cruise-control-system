import argparse
import glob
import os

parser = argparse.ArgumentParser(description='Formattinf image filename')
parser.add_argument('--file_extension', default="jpg", type=str, help='Image file extension')

if __name__ == '__main__':
    args = parser.parse_args()
    fileExtension = args.file_extension
    images = glob.glob('*.' + fileExtension)
    for i, image in enumerate(images):
        os.rename(image, "image" + str(i + 1) + "." + fileExtension)