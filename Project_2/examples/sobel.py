import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from matplotlib.widgets import Slider, Button
from examples.line_polinom import *

cap = cv2.VideoCapture('../harder_challenge_video.mp4')

re, image = cap.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)





########### ALGORITHM ######################

def sobel_x_binary(im_CV1, ksize=3, th=(0,255)):
    sobx = np.abs(cv2.Sobel(im_CV1, cv2.CV_64F, 1, 0, ksize=ksize))
    sobx = np.uint8(255 * sobx / np.max(sobx))
    mask = np.zeros_like(sobx)
    mask[(sobx > th[0]) & (sobx <= th[1])] = 1
    return mask


def sobel_y_binary(im_CV1, ksize=3, th=(0, 255)):
    sobx = np.abs(cv2.Sobel(im_CV1, cv2.CV_64F, 0, 1, ksize=ksize))
    sobx = np.uint8(255 * sobx / np.max(sobx))

    mask = np.zeros_like(sobx)
    mask[(sobx > th[0]) & (sobx <= th[1])] = 1
    return mask

def mag_thresh_binary(im, sobel_kernel=3, mag_thresh=(0, 255)):

    sobx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    soby = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sob = np.sqrt(sobx**2 + soby**2)
    sob = np.uint8(255*sob / np.max(sob))

    ret = np.zeros_like(sob)
    ret[(sob>mag_thresh[0]) & (sob < mag_thresh[1])] = 1
    return ret

def filter_color(im, th=(0,255)):
    mask = np.zeros_like(im)

    mask[(im > th[0]) & (im <= th[1])] = 1
    return mask

def warp(im):
    imshape = im.shape
    src = np.float32([[350, 670], [581, 470], [719, 470], [1000, 670]])

    min_x = int(imshape[1] * 0.20)
    max_x = int(imshape[1] * 0.80)
    min_y = int(imshape[0] * 0.07)
    max_y = int(imshape[0] * 0.93)
    dst = np.float32([[min_x, max_y], [min_x, min_y], [max_x, min_y], [max_x, max_y]])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(im, M, imshape[1::-1])
    return warped

def abs_sobel_thresh(img, sobel_kernel=3, th=(0, 255), th_col=(0,255), mag=True):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # im = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    h_channel = hls[:,:,0]
    l_channel = hls[:, :,1]
    s_channel = hls[:, :,2]

    # soby = np.abs(cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    #
    # angls = np.arctan2(sobx, soby)
    #
    # ret = np.zeros_like(angls)
    # ret[(angls>mag_thresh[0]) & (angls <= mag_thresh[1])] = 1
    mask1 = filter_color(s_channel, th=th_col)
    if mag:
        mask2 = mag_thresh_binary(l_channel, mag_thresh=th)
    else:
        mask2 = sobel_x_binary(l_channel, th=th, ksize=sobel_kernel)
    mask3 = filter_color(l_channel, th=(126, 255))
    mask = np.zeros_like(mask1)
    mask[((mask1 == 1) | (mask2 == 1)) & (mask3 == 1)] = 1

    return  fit_polynomial(warp(mask))

############################################################



# Run the function
grad_binary = abs_sobel_thresh(image,sobel_kernel=3, th=(0,255))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))


def upfate(val):
    grad_binary = abs_sobel_thresh(image,sobel_kernel=3, th=(tresh_min.val,tresh_max.val),
                                   th_col=(tresh_min_color.val, tresh_max_color.val))

    ax2.imshow(grad_binary, cmap='gray')
    ax1.imshow(image)
    f.canvas.draw_idle()

def change_picture(d):
    global cap
    global image
    ret, im = cap.read()
    if ret:
        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    upfate(0)
    print("yololo")

axcolor = 'lightgoldenrodyellow'
sl_min = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
tresh_min = Slider(sl_min, "Thresh_min", 0, 255, valinit=56, valstep=1)
sl_max = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
tresh_max = Slider(sl_max, "Thresh_max", 0, 255, valinit=255, valstep=1)

sl_min_col = plt.axes([0.25, 0.9, 0.65, 0.03], facecolor=axcolor)
tresh_min_color = Slider(sl_min_col, "Color_min", 0, 255, valinit=31, valstep=1)
sl_max_col = plt.axes([0.25, 0.95, 0.65, 0.03], facecolor=axcolor)
tresh_max_color = Slider(sl_max_col, "Color_max", 0, 255, valinit=255, valstep=1)

axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
bcut = Button(axcut, 'NEXT', color='red', hovercolor='green')
bcut.on_clicked(change_picture)

tresh_max_color.on_changed(upfate)
tresh_min_color.on_changed(upfate)
tresh_min.on_changed(upfate)
tresh_max.on_changed(upfate)
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()