import itertools
import shutil
import numpy as np
import os
import sys
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image


ROOT_DIR = '../immatriculation_plate/'
DATA_DIR = ROOT_DIR + 'image/'
PLATE_DIR = DATA_DIR + 'plate/'
BGS_DIR = DATA_DIR + 'background/'
TRAIN_IMGS_DIR = DATA_DIR + 'GENERATED_DATASET/train/images/'
TRAIN_ANNS_DIR = DATA_DIR + 'GENERATED_DATASET/train/labels/'
TEST_IMGS_DIR = DATA_DIR + 'GENERATED_DATASET/valid/images/'
TEST_ANNS_DIR = DATA_DIR + 'GENERATED_DATASET/valid/labels/'
DEFAULT_POSITION = 0 # Une constante définissant l'absence de plaque dans l'image
NOISE_SCALE = 0.05 # Le rapport de bruit(par rapport à 1) : Pratiquement entre 0.01 et 0.1
MAX_IMAGE = 10000 # Nombre d'images à générer
OUTPUT_SHAPE = (400, 400)
Visual = False #visualisation
DIRS = [TRAIN_IMGS_DIR, TRAIN_ANNS_DIR, TEST_ANNS_DIR,TEST_IMGS_DIR]


def matrice_rotation(yaw, pitch, roll):
    
    # Rotation selon l'axe Y
    c, s = np.cos(yaw), np.sin(yaw)
    M = np.matrix([   [  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])
    
    # Rotation selon l'axe X
    c, s = np.cos(pitch), np.sin(pitch)
    M = np.matrix([   [ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M
    
    # Rotation selon l'axe Z
    c, s = np.cos(roll), np.sin(roll)
    M = np.matrix([   [  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M
    return M

def bounding_box(mask, default = DEFAULT_POSITION, output = OUTPUT_SHAPE):  
    top = default
    bottom = default
    mask=mask[:,:,0]
    for i in range(0,output[0],1):
        if top == default and any(mask[i] > 0) :
            top = i
        if bottom == default and any(mask[output[0] - 1 - i] > 0) :
            bottom = output[0] - 1 - i
        if top != default and bottom != default :
            break
    left = default
    right = default
    for i in range(0,output[1],1):
        if left == default and any(mask.T[i] > 0) :
            left = i
        if right == default and any(mask.T[output[1] - 1 - i] > 0) :
            right = output[1] - 1 - i
        if left != default and right != default :
            break
    
    return top, left, bottom, right

def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = BGS_DIR + '{:08d}.jpg'.format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname)
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY) / 255
       
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]
    return bg

def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    from_size = np.array([[from_shape[1], from_shape[0]]]).T 
    to_size = np.array([[to_shape[1], to_shape[0]]]).T 

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)

    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    else:
        out_of_bounds = False

    roll = random.uniform(-np.pi / 6, np.pi / 6) * rotation_variation
    pitch = random.uniform(-np.pi / 6, np.pi / 6) * rotation_variation
    yaw = random.uniform(-np.pi / 6, np.pi / 6) * rotation_variation
    
    M = matrice_rotation(yaw, pitch, roll)[:2, :2]
    
    h, w, d= from_shape
    corners = np.matrix([[-w, +w, -w, +w],
                         [-h, -h, +h, +h]]) / 2
    
    skewed_size = np.array(np.max(M * corners, axis=1) - np.min(M * corners, axis=1))
    scale *= np.min(to_size / skewed_size)
    M *= scale
    trans = (np.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans
    center_to = to_size / 2.
    center_from = from_size / 2.
    M = np.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds 

def get_random_file(directory):
    files = os.listdir(directory)
    if len(files) == 0:
        return None
    else:
        return os.path.join(directory, random.choice(files))

def generate_img(num_bg_images):
    bg = generate_bg(num_bg_images)
    plate_path=get_random_file(PLATE_DIR)
    plate_name = plate_path.split('/')[-1].split('.')[0]
    plate= plt.imread(plate_path)
    plate=plate[:,:,:]
    fig=plt.figure()
    mask=np.ones(plate.shape)

    if Visual:
        fig.add_subplot(2, 3, 1)
        plt.imshow(plate,cmap='gray')
        fig.add_subplot(2, 3, 4)
        plt.imshow(bg,cmap='gray')
    
    M, out_of_bounds = make_affine_transform(from_shape=plate.shape,
                                             to_shape=bg.shape,
                                             min_scale=0.3,
                                             max_scale=1.0,
                                             rotation_variation=0.8,
                                             scale_variation=1.0,
                                             translation_variation=1.0)
    
    if out_of_bounds:
        return generate_img(num_bg_images)
    
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    if Visual:
        fig.add_subplot(2, 3, 2)
        plt.imshow(plate,cmap='gray')
        
    mask = cv2.warpAffine(mask, M, (bg.shape[1], bg.shape[0]))
    if Visual:
        fig.add_subplot(2, 3, 3)
        plt.imshow(mask,cmap='gray')
    
    box = bounding_box(mask)

    img = plate[:,:,0] * mask[:,:,0] + bg[:,:]* (1.0 - mask[:,:,0]) 

    if Visual:
        fig.add_subplot(2, 3, 5)
        plt.imshow(img,cmap='gray')
    
    img += np.random.normal(scale = NOISE_SCALE, size = img.shape)
    if Visual:
        fig.add_subplot(2, 3, 6)
        plt.imshow(img,cmap='gray')
        plt.show()
    plt.close()
  
    return img, not out_of_bounds,plate_name, box

def generate_imgs():
    num_bg_images = len(os.listdir(BGS_DIR))
    while True:
        yield generate_img(num_bg_images)


def save_file(img_idx, plate, plate_number, box , t):
    if t=='train':
        img_name = TRAIN_IMGS_DIR + '{:08d}_{}.png'.format(img_idx, plate_number)
        ann_name = TRAIN_ANNS_DIR + '{:08d}_{}.txt'.format(img_idx, plate_number)
    elif t =='test':
        img_name = TEST_IMGS_DIR + '{:08d}_{}.png'.format(img_idx, plate_number)
        ann_name = TEST_ANNS_DIR + '{:08d}_{}.txt'.format(img_idx, plate_number)

    ann = open(ann_name, mode = 'w', encoding = 'utf-8')
    contenu = '{} {} {} {} {}\n'.format(1,*box)
    ann.write(contenu)
    ann.close()
    cv2.imwrite(img_name, plate * 255)

if __name__=="__main__":
    # Création des chemins des données
    for DIR in DIRS:
        shutil.rmtree(DIR, ignore_errors=True)
        os.makedirs(DIR)
    

    im_gen = itertools.islice(generate_imgs(),MAX_IMAGE)
    for img_idx, (plate, p,plate_number, box) in enumerate(im_gen):
        if (img_idx%5==0):
            save_file(img_idx, plate, plate_number, box , t='test')
        else:
            save_file(img_idx, plate, plate_number, box , t='train')

        i = (img_idx / MAX_IMAGE) * 100
        if int(i) == i:
            sys.stdout.write("\r%d%%" % i)
            sys.stdout.flush()

    print('\rLa création de {} données est términé !!'.format(MAX_IMAGE))