from PIL import Image
import os
import numpy as np
import cv2
import pandas as pd

###parameters to adapt to the database
nbr_horizons = 15207
nbr_profile = 3101
i_h, i_w = 1500,1500  ##wanted high and width of the pictures and masks

###paths
path_excel = r"D:\Data\protat.a\data_horizons.xlsx"
path_pictures = r"J:\NUTZER\Protat.A\Daten\PROFILBILDER_Y - Kopie"
path_images = r"D:\Data\protat.a\images"
path_masks = r"D:\Data\protat.a\masks"


###importation of Excel data
data = pd.read_excel(path_excel)
numero = data["PointID"]
lettre = data["Bundesland"]
symboles = data["Horizontsymbol"]
hauteur_inf = data["Untergrenze"]
depthmax = data["max_depth"]
prof_hor = data["Point"]

### extract the main symbol from horizons names

symboles_simples = []
for k in range(nbr_horizons):
    symb = ""
    horizon = symboles[k]
    horizon = str(horizon)
    for car in horizon:
        if car in "AGSBCHMOP":
            symb = car
    if symb == "":
        symboles_simples += "0"
    else:
        symboles_simples += str(symb)


#### creation list of profiles
profile = []
for k in range(nbr_profile):
    if numero[k] < 10:
        profile += [str(str(lettre[k]) + "_000" + str(numero[k]))]
    elif numero[k] < 100:
        profile += [str(str(lettre[k]) + "_00" + str(numero[k]))]
    elif numero[k] < 1000:
        profile += [str(str(lettre[k]) + "_0" + str(numero[k]))]
    else:
        profile += [str(str(lettre[k]) + "_" + str(numero[k]))]



###creation of a list with the real depth of the bottom of the A-horizon

depth_reelle = []
k = 0
for i in range(nbr_profile):
    d = 0
    profil = int(numero[i])
    while prof_hor[k] == profil:
        if symboles_simples[k] == "A" and symboles_simples[k+1] != "A":
            d = hauteur_inf[k]
        k += 1
    if d == 0:
        depth_reelle += [str(profile[i] + "= 00")]
    elif d < 10:
        depth_reelle += [str(profile[i] + "= 0" + str(int(d)))]
    else:
        depth_reelle += [str(profile[i] + "= " + str(int(d)))]

depth_reelle.sort() #sorted by alphabetic order to ease the correlation between the 2 lists

hauteur_image = []
for elt in depth_reelle:
    hauteur_image += [int(elt[-3:])] #get only the depth without the profile name


###list of the depths in the resized pictures
depths = []
for k in range(nbr_profile):
    depth = ((hauteur_image[k] * i_h)/depthmax[k])
    depths.append(depth)


####list images
image_dataset = os.listdir(path_pictures)
images = []
for profile in image_dataset:
    path = os.path.join(path_pictures, profile)
    img = Image.open(path).convert('RGB')
    images.append(img)


###resize images and convert into array
y = np.zeros((nbr_profile, i_h, i_w, 1), dtype=np.int32)
X = np.zeros((nbr_profile, i_h, i_w,3), dtype=np.float32)

for i in range(nbr_profile) :
    single_img = images[i]

    single_img = single_img.resize((i_h, i_w))
    single_img = np.reshape(single_img, (i_h, i_w, 3))

    X[i] = single_img


### create masks (that has the same dimensions of the image where each pixel is valued at 0 except for the A-horizon where pixel value = 1)

for k in range (nbr_profile):
    mask = np.zeros((i_h, i_w, 1), dtype=np.int32)
    lim = depths[k]

    if lim == 0:  #if depth =0 : no A-horizon in the profile
        y[k] = mask
    else:
        cv2.rectangle(mask, (0, 0), (i_w, int(lim)), 1, -1)    # assimilate horizon to a rectangle
        y[k] = mask

###save mask and images as png
i = 1
j = 1

for img in X:
    cv2.imwrite(path_images + "image" + str(i) + ".png", img)
    i += 1

for img in y:
    cv2.imwrite(path_masks + "mask" + str(j) + ".png", img)
    j += 1
