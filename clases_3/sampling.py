# https://stackoverflow.com/questions/53551410/how-to-randomly-select-images-and-put-them-to-multiple-folders-in-python

import os, random
import shutil

m = 50
n = 150

src_dir = "/home/jduran/master-bigData/datos/Modelo 2/TRAIN/T/"
dst_dir = "/home/jduran/master-bigData/datos/Modelo 2/Sampling/T"

file_list = os.listdir(src_dir)

for i in range(m):
    for j in range(n):
        a = random.choice(file_list)
        #file_list.remove(a)
        shutil.copy(src_dir + a, dst_dir + "/")