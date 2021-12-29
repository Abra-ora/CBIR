# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 23:00:27 2021

@author: Ab_Amn
"""
import xlwt
from xlwt import Workbook
import cv2
import numpy as np
import os
import glob
import time
import xlrd
import csv
# import xlwings as xw
from PIL import Image
# from skimage.feature import graycomatrix

tic = time.time()


def coarseness(image, kmax):
	image = np.array(image)
	w = image.shape[0]
	h = image.shape[1]
	kmax = kmax if (np.power(2, kmax) < w) else int(np.log(w) / np.log(2))
	kmax = kmax if (np.power(2, kmax) < h) else int(np.log(h) / np.log(2))
	average_gray = np.zeros([kmax, w, h])
	horizon = np.zeros([kmax, w, h])
	vertical = np.zeros([kmax, w, h])
	Sbest = np.zeros([w, h])

	for k in range(kmax):
		window = np.power(2, k)
		for wi in range(w)[window:(w-window)]:
			for hi in range(h)[window:(h-window)]:
				average_gray[k][wi][hi] = np.sum(
				    image[wi-window:wi+window, hi-window:hi+window])
		for wi in range(w)[window:(w-window-1)]:
			for hi in range(h)[window:(h-window-1)]:
				horizon[k][wi][hi] = average_gray[k][wi+window][hi] - \
				    average_gray[k][wi-window][hi]
				vertical[k][wi][hi] = average_gray[k][wi][hi+window] - \
				    average_gray[k][wi][hi-window]
		horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
		vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

	for wi in range(w):
		for hi in range(h):
			h_max = np.max(horizon[:, wi, hi])
			h_max_index = np.argmax(horizon[:, wi, hi])
			v_max = np.max(vertical[:, wi, hi])
			v_max_index = np.argmax(vertical[:, wi, hi])
			index = h_max_index if (h_max > v_max) else v_max_index
			Sbest[wi][hi] = np.power(2, index)

	fcrs = np.mean(Sbest)
	return fcrs


def contrast(image):
	image = np.array(image)
	image = np.reshape(image, (1, image.shape[0]*image.shape[1]))
	m4 = np.mean(np.power(image - np.mean(image), 4))
	v = np.var(image)
	std = np.power(v, 0.5)
	alfa4 = m4 / np.power(v, 2)
	fcon = std / np.power(alfa4, 0.25)
	return fcon


def directionality(image):
	image = np.array(image, dtype='int64')
	h = image.shape[0]
	w = image.shape[1]
	convH = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
	convV = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
	deltaH = np.zeros([h, w])
	deltaV = np.zeros([h, w])
	theta = np.zeros([h, w])

	# calc for deltaH
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaH[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convH))
	for wi in range(w)[1:w-1]:
		deltaH[0][wi] = image[0][wi+1] - image[0][wi]
		deltaH[h-1][wi] = image[h-1][wi+1] - image[h-1][wi]
	for hi in range(h):
		deltaH[hi][0] = image[hi][1] - image[hi][0]
		deltaH[hi][w-1] = image[hi][w-1] - image[hi][w-2]

	# calc for deltaV
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaV[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convV))
	for wi in range(w):
		deltaV[0][wi] = image[1][wi] - image[0][wi]
		deltaV[h-1][wi] = image[h-1][wi] - image[h-2][wi]
	for hi in range(h)[1:h-1]:
		deltaV[hi][0] = image[hi+1][0] - image[hi][0]
		deltaV[hi][w-1] = image[hi+1][w-1] - image[hi][w-1]

	deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
	deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

	# calc the theta
	for hi in range(h):
		for wi in range(w):
			if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
				theta[hi][wi] = 0;
			elif(deltaH[hi][wi] == 0):
				theta[hi][wi] = np.pi
			else:
				theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
	theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

	n = 16
	t = 12
	cnt = 0
	hd = np.zeros(n)
	dlen = deltaG_vec.shape[0]
	for ni in range(n):
		for k in range(dlen):
			if((deltaG_vec[k] >= t) and (theta_vec[k] >= (2*ni-1) * np.pi / (2 * n)) and (theta_vec[k] < (2*ni+1) * np.pi / (2 * n))):
				hd[ni] += 1
	hd = hd / np.mean(hd)
	hd_max_index = np.argmax(hd)
	fdir = 0
	for ni in range(n):
		fdir += np.power((ni - hd_max_index), 2) * hd[ni]
	return fdir


def roughness(fcrs, fcon):
	return fcrs + fcon


"""
def linelikeness(image):
    #  TODO: Check this functions working. If it works properly or not
    #  Possible Runtime warnings

    H, W = image.shape[:2]

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    img_prewittx = cv2.filter2D(image, -1, kernelx)
    img_prewitty = cv2.filter2D(image, -1, kernely)
    angles = np.arctan(img_prewitty / img_prewittx) + (np.pi / 2)

    n_bins = 9
    bin_angles = np.array(range(0, 180, 20)) * np.pi / 180

    digitized_angles = np.digitize(angles, bin_angles)

    comat = graycomatrix(digitized_angles, [1], [0, np.pi/2], levels=256)

    line_likeness = 0
    for i in range(n_bins):
        for j in range(n_bins):
            line_likeness += comat[i, j] * np.cos((i-j) * 2 * np.pi / n_bins)
    line_likeness /= np.sum(comat)

    return line_likeness
"""


def write_in_file(path_bd, path_xlrd):

        # Workbook is created
        wb = Workbook()
        s = wb.add_sheet('Sheet 1')
        s.write(0, 0, 'pic_name')
        s.write(0, 1, 'Coarseness')
        s.write(0, 2, 'Contrast')
        s.write(0, 3, 'Directionality')
        s.write(0, 4, 'Roughness')
        s.write(0, 5, 'Total')
       # normalisation :
        s.write(0, 6, 'N_Coarseness')
        s.write(0, 7, 'N_Contrast')
        s.write(0, 8, 'N_Directionality')
        s.write(0, 9, 'N_Roughness')

        # load the training dataset
        train_path = path_bd
        train_names = os.listdir(train_path)

        # loop over the training dataset
        cur_path = os.path.join(train_path, '*g')
        img_name = train_names
        i = 0

        for file in glob.glob(img_name):

            # print('For image {} named {}:'.format(i+1,img_name[i]))

            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            fcrs = coarseness(img, 5)
            fcon = contrast(img)
            fdir = directionality(img)
            f_r = roughness(fcrs, fcon)
            total = fcrs + fcon + fdir + f_r
            s.write(i+1, 0, img_name[i])
            s.write(i+1, 1, fcrs)
            s.write(i+1, 2, fcon)
            s.write(i+1, 3, fdir)
            s.write(i+1, 4, f_r)
            s.write(i+1, 5, total)
            # Normalised_value
            s.write(i+1, 6, fcrs/total)
            s.write(i+1, 7, fcon/total)
            s.write(i+1, 8, fdir/total)
            s.write(i+1, 9, f_r/total)
            i += 1

        wb.save(path_xlrd)

#############################################


def get_similarity(image, file_xlrd, max_images):

    wb = xlrd.open_workbook(file_xlrd)
    s = wb.sheet_by_index(0)

    v1 = vector(image)
    num_rows = s.nrows

    curr_row = 1
    x = {}
    while curr_row < num_rows:

         crs = s.cell_value(curr_row, 1)
         con = s.cell_value(curr_row, 2)
         drc = s.cell_value(curr_row, 3)
         rog = s.cell_value(curr_row, 4)
         v2 = np.array([crs, con, drc, rog])
        # d = get_canberra_distance(v1,v2)
         x[s.cell_value(curr_row, 0)] = get_canberra_distance(v1, v2)
         x = dict(sorted(x.items(), key=lambda item: item[1]))
         last = list(x.items())[:max_images]

         curr_row += 1

    return last


#############################################
def vector(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	coarseness_ = coarseness(img,5)
	contrast_ = contrast(img)
	directionality_ = directionality(img)
	roughness_ = roughness(coarseness(img, 5),contrast(img))
	return np.array([coarseness_, contrast_, directionality_, roughness_])




def get_canberra_distance(v1, v2):
		return np.round(np.sum(np.fabs(v1 - v2) / (np.fabs(v1) + np.fabs(v2))), 4)



# if __name__ == '__main__':
  
    
   
#     file_xlrd = ("features_2.xls") 
    
# img1 = cv2.imread('./coil-100/obj1__0.png')
# img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)


# path_bd = './coil-100'
# file = './feature.xls'
# # write_in_file(path_bd,file)
# # print(get_similarity(img1, file, 5))
# a = dict(get_similarity(img1, file, 10))
# for key in a:
#     print(key)
#     im = Image.open('./coil-100/{}'.format(key))
#     im.show()
# toc = time.time()
# print("Computation time is {} minutes".format((toc-tic)/60))

