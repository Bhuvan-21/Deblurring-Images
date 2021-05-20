import matplotlib.pyplot as plt
import numpy as np
import cv2

def infer(model, inp):
    
    model.eval()
    out = model(inp)
    out = out.detach()
    return out

def rescale(t):
    
    t = t + 1
    t = t * 127.5
    return t

def scale(x, feature_range = (1, -1)):
    
    a, b = feature_range
    x /= 255.0
    x = (a-b)*x + b
    
    return x


def imsave(img, path_o):
    img = img.numpy()
    img = np.ascontiguousarray(img, dtype=np.uint8)
    img = np.transpose(img, (1, 2, 0))
    plt.imsave(path_o, img)




def detect_face(img):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
    
    faces = []
    for (x, y, w, h) in face_rects:
        margin_y = h//10 #10% margin on each side
        margin_x = w//10 #10% margin on each side
        faces.append(img[y-margin_y:y+h+margin_y, x-margin_x:x+w+margin_x, :])
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 255, 255), max((w*h//50000), 1))
        
    return faces