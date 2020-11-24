import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from glob import glob
import cv2
from scipy import misc
import imageio
from sklearn.decomposition import PCA

if __name__ == '__main__':
    covid_plots_dir = 'output/exp_covid_vowel_section_1_with_melplots/plots_phasor_covid/*'
    normal_plots_dir = 'output/exp_covid_vowel_section_1_with_melplots/plots_phasor_normal/*'

    covid_filenames = glob(covid_plots_dir)
    normal_filenames = glob(normal_plots_dir)

    covid_images = np.zeros((87, 480, 640))
    for i, file_name in enumerate(covid_filenames):
        image = imageio.imread(file_name)
        gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
        gray = gray(image)  
        covid_images[i,:,:] = gray
        # plt.imshow(gray, cmap = plt.get_cmap(name = 'gray'))
        # plt.show()

    normal_images = np.zeros((87, 480, 640))
    for i, file_name in enumerate(normal_filenames):
        image = imageio.imread(file_name)
        gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
        gray = gray(image)
        normal_images[i, :, :] = gray
    
    pca_covid = PCA(22)
    X_proj_covid = pca_covid.fit_transform(covid_images.reshape(-1, 480*640))

    #Setup a figure 8 inches by 8 inches 
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05) 
    # plot the faces, each image is 64 by 64 pixels
    for i in range(21): 
        ax = fig.add_subplot(7, 3, i+1, xticks=[], yticks=[]) 
        ax.imshow(np.reshape(pca_covid.components_[i,:], (480,640)), cmap=plt.cm.bone, interpolation='nearest')
    plt.title('Covid data')

    pca_normal = PCA(22)
    X_proj_normal = pca_normal.fit_transform(normal_images.reshape(-1, 480*640))

    #Setup a figure 8 inches by 8 inches 
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05) 
    # plot the faces, each image is 64 by 64 pixels
    for i in range(21): 
        ax = fig.add_subplot(7, 3, i+1, xticks=[], yticks=[]) 
        ax.imshow(np.reshape(pca_normal.components_[i,:], (480,640)), cmap=plt.cm.bone, interpolation='nearest')
    plt.title('Normal data')
    plt.show()

    print('Works till here!')