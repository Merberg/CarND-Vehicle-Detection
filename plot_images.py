
# coding: utf-8

# # Plotting Functions
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import operator
import random
get_ipython().magic('matplotlib inline')

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
FIGURE_SIZE = (24,12)
FONT_SIZE = 18

def plotOne(img1, title1, cmap1=None):
    plt.imshow(img1, cmap=cmap1)
    plt.title(title1)
    plt.show()
    return

def plotMany(nrows, ncols, images, titles, cmaps):
    f, axes = plt.subplots(nrows, ncols, figsize=FIGURE_SIZE)
    if nrows == 1 or ncols == 1:
        for idx, img in enumerate(images):
            axes[idx].imshow(img, cmap=cmaps[idx])
            axes[idx].set_title(titles[idx], fontsize=FONT_SIZE) 
    else:
        c = 0
        for idx, img in enumerate(images):
            r = idx//ncols
            axes[r,c].imshow(img, cmap=cmaps[idx])
            axes[r,c].set_title(titles[idx], fontsize=FONT_SIZE)
            c = (idx + 1) % ncols        
    plt.tight_layout()
    plt.show()
    return

def plotTwo(img1, img2, title1, title2, cmap1=None, cmap2=None):
    plotMany(1, 2, [img1,img2], [title1,title2], [cmap1,cmap2])
    return

def plotWithVerticles(ncols, images, offset, titles, cmaps):
    f, axes = plt.subplots(1, ncols, figsize=FIGURE_SIZE)
    for idx, img in enumerate(images):
        axes[idx].imshow(img, cmap=cmaps[idx])
        axes[idx].set_title(titles[idx], fontsize=FONT_SIZE)
        xL = offset
        xR = IMAGE_WIDTH-offset
        axes[idx].plot((xL, xL), (0, IMAGE_HEIGHT), 'r-')
        axes[idx].plot((xR, xR), (0, IMAGE_HEIGHT), 'r-')
    plt.show()
    return

def plotCreateSecondOrderXY(leftCoeff, rightCoeff):
    y = np.linspace(0, IMAGE_HEIGHT-1, IMAGE_HEIGHT)
    xL = leftCoeff[0]*y**2 + leftCoeff[1]*y + leftCoeff[2]
    xR = rightCoeff[0]*y**2 + rightCoeff[1]*y + rightCoeff[2]
    return xL, xR, y

def plotWithSecondOrder(img, leftCoeff, rightCoeff, title):
    xL, xR, y = plotCreateSecondOrderXY(leftCoeff, rightCoeff)
    plt.imshow(img, cmap='gray')
    plt.plot(xL, y, color='yellow')
    plt.plot(xR, y, color='yellow')
    plt.ylim(IMAGE_HEIGHT, 0)
    plt.xlim(0, IMAGE_WIDTH)
    plt.title(title)
    plt.show()
    return

def plotTwoWithFirstFilled(img1, img2, leftCoeff, rightCoeff, title1, title2):
    xL, xR, y = plotCreateSecondOrderXY(leftCoeff, rightCoeff)  
    L = leftCoeff[0]*IMAGE_HEIGHT**2 + leftCoeff[1]*IMAGE_HEIGHT + leftCoeff[2]
    R = rightCoeff[0]*IMAGE_HEIGHT**2 + rightCoeff[1]*IMAGE_HEIGHT + rightCoeff[2]
    verts = [(L,0)] + list(zip(xL, y)) + list(zip(reversed(xR), reversed(y))) + [(R,0)]
    poly = patches.Polygon(verts, facecolor='green', edgecolor='yellow')
    
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(title1, fontsize=FONT_SIZE)
    axes[0].add_patch(poly)
    
    axes[1].imshow(img2)
    axes[1].set_title(title2, fontsize=FONT_SIZE)
    
    plt.ylim(IMAGE_HEIGHT, 0)
    plt.xlim(0, IMAGE_WIDTH)
    plt.show() 
    return

