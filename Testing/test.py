import sys 
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models import Generator, Generator_R
from utils import scale, rescale, imsave, infer, detect_face
    

args = sys.argv

if len(args) < 2:
    print("Wrong input format, example : ")
    print("python test.py input_image_path")
    exit()

path_i = args[1]

G = Generator(32)
GR = Generator_R(32)

G.load_state_dict(torch.load("Checkpoints/G_22.pt", map_location=torch.device('cpu')))
GR.load_state_dict(torch.load("Checkpoints/GR_7.pt", map_location=torch.device('cpu')))

img = plt.imread(path_i)
faces = detect_face(img)

if len(faces) == 0:
    print("No faces detected")
    exit()

resz = cv2.resize(faces[0], (100, 100))
plt.imsave("out_ld.png", resz)

resz = resz.reshape(1, 100, 100, 3)
resz = np.transpose(resz, (0, 3, 1, 2))
resz = torch.from_numpy(resz)
resz = resz.float()
inp = scale(resz)
out1 = infer(G, inp)
out2 = infer(GR, inp)


inp = rescale(inp)
out1 = rescale(out1)
out2 = rescale(out2)

w = 0.5
out = out1 * w + out2 * (1 - w)
imsave(out[0], "out_hd.png")

