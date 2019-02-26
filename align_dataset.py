from glob import glob
from tqdm import tqdm
import os
from shutil import rmtree
import numpy as np
from scipy.misc import imsave, imread

from .new_alignment_utils import align_image

BASE_DIR = '/home/nazar/Downloads/UTKFace dataset/original'
SAVE_DIR = '/home/nazar/Downloads/UTKFace dataset/aligned'
TARGET_SIZE = 128


def parse_landmarks():
    def read_txt(path):
        res = {}
        with open(path, 'r') as f:
            data = f.readlines()
        for line in data:
            line = line.split(' ')
            res[line[0]] = np.array(line[1:-1], dtype=np.float64).reshape([-1, 2])
        return res
    return {**read_txt(os.path.join(BASE_DIR, 'landmark_list_part1.txt')),
            **read_txt(os.path.join(BASE_DIR, 'landmark_list_part2.txt')),
            **read_txt(os.path.join(BASE_DIR, 'landmark_list_part3.txt'))}


def align_image_ans_save(args):
    try:
        path, lan, image_target_size, image_top_margin, save_path = args
        imag = imread(path)
        face, params = align_image(imag, lan, image_target_size, image_top_margin)
        imsave(save_path, face)
    except Exception as e:
        print('\n fuck!!!', path, e)


def main():
    if os.path.isdir(SAVE_DIR):
        rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    paths_img = sorted(glob(os.path.join(BASE_DIR, 'UTKFace', '*.jpg')))
    landmarks_dict = parse_landmarks()
    args = []
    for path in tqdm(paths_img):
        try:
            landmarks = landmarks_dict[os.path.basename(path[:-9])]
            save_path = os.path.join(SAVE_DIR, os.path.basename(path))
            args.append((path, landmarks, TARGET_SIZE, 0.07, save_path))
        except Exception:
            print(path)
    a = [align_image_ans_save(arg) for arg in args]
    print('nonono')




if __name__ == "__main__":
    main()