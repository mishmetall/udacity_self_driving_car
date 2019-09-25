import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from matplotlib.widgets import Slider, Button
from line_polinom import *
from polinom2 import *
import pickle

with open("../undist.pkl", "rb") as f:
    data = pickle.load(f)

mtx = data["mtx"]
dist = data["dist"]

# Read in an image and grayscale it
# image = mpimg.imread('/home/sviatoslavpovod/Documents/udacity/CarND-Advanced-Lane-Lines/test_images/test2.jpg')

cap = cv2.VideoCapture("../../challenge_video.mp4")







########### ALGORITHM ######################

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 770  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # x = Ay^2+By+C
    # x * xm_per_pix = A (y*ym_per_pix) ^ 2 + B * y * ym_per_pix + C

    left_curverad = np.power(1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2, 1.5) / np.abs(
        2 * left_fit_cr[0])
    right_curverad = np.power(1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2, 1.5) / np.abs(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


def sobel_x_binary(im_CV1, ksize=3, th=(0,255),bin=True):
    sobx = np.abs(cv2.Sobel(im_CV1, cv2.CV_64F, 1, 0, ksize=ksize))
    sobx = np.uint8(255 * sobx / np.max(sobx))
    if not bin:
        return sobx
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

def grad_direction(im, sobel_kernel=(3,3), th=(-np.pi,np.pi)):


    soby = np.abs(cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    sobx = np.abs(cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=sobel_kernel))

    angls = np.arctan2(soby, sobx)

    ret = np.zeros_like(angls)
    ret[(angls>th[0]) & (angls <= th[1])] = 1
    return ret

def warp(im, inverse=False):
    imshape = im.shape

    if not hasattr(warp, "M"):
        src = np.float32([[350, 670], [581, 470], [719, 470], [1000, 670]])

        min_x = int(imshape[1] * 0.20)
        max_x = int(imshape[1] * 0.80)
        min_y = int(imshape[0] * 0.07)
        max_y = int(imshape[0] * 0.93)
        dst = np.float32([[min_x, max_y], [min_x, min_y], [max_x, min_y], [max_x, max_y]])

        warp.M = cv2.getPerspectiveTransform(src, dst)
        warp.Minv = cv2.getPerspectiveTransform(dst, src)

    if inverse:
        warped = cv2.warpPerspective(im, warp.Minv, imshape[1::-1])
    else:
        warped = cv2.warpPerspective(im, warp.M, imshape[1::-1])

    return warped

def normalize(im_cv1):
    ar = np.asarray(im_cv1, dtype=np.float64) / np.mean(im_cv1)
    return np.uint8(255 * ar / np.max(ar))

def illumination_correction(image):
    ### Convert color
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    l_channel = hls[:, :, 1]

    # apply the CLAHE algorithm to the L channel
    clahe = cv2.createCLAHE()
    clahe.setClipLimit(4)

    hls[:, :, 1] = clahe.apply(l_channel)

    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

def color_grad_binary(img, sobel_kernel=3, th=(0, 255), th_col=(0,255), th_ang=(-np.pi,np.pi), mag=False):

    ### Convert color
    # img = illumination_correction(img)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    h_channel = hls[:,:,0]
    l_channel = hls[:, :,1]
    s_channel = hls[:, :,2]

    s_channel = normalize(s_channel)
    ## Good way to find yellow line in s_channel
    mask1 = sobel_x_binary(s_channel, th=th, ksize=11, bin=True)
    # mask11 = grad_direction(s_channel, sobel_kernel=11, th=(0.7, 1.3))
    mask1 = filter_color(s_channel, th=(np.std(s_channel)+th_col[0], 255))

    ## Find white lines by applying gradient technique
    if mag:
        mask2 = mag_thresh_binary(l_channel, mag_thresh=th)
    else:
        mask2 = sobel_x_binary(l_channel, th=th_ang, ksize=9)


    ## In challenge_video there are black line with good gradients, i want to remove it
    mask3 = filter_color(l_channel, th=(126, 255))
    mask3 = cv2.GaussianBlur(mask3*255, ksize=(3,3), sigmaX=1)
    mask3[mask3 < 255] = 0
    mask3 = np.uint8(mask3/255.)


    ### This is tests with direction of gradient, but i dont use it now
    # mask4 = grad_direction(l_channel, sobel_kernel=17, th=(th_ang[0], th_ang[1]))
    # opening = cv2.morphologyEx(mask4, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    ## Finilize results of gradien/color thresholds
    mask = np.zeros_like(mask1)
    mask[  ((mask2 == 1) | (mask1==1)) & (mask3==1)] = 1

    return  mask
############################################################
                ##VIZUALIZER#
############################################################



# def run_matplot(cap):
re, image = cap.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def upfate(val):
    global left_coef
    global right_coef
    binary = color_grad_binary(image,sobel_kernel=11, th=(tresh_min.val,tresh_max.val),
                                   th_col=(tresh_min_color.val, tresh_max_color.val),
                                   th_ang=(tresh_min_ang.val, tresh_max_ang.val))
    left_coef, right_coef, left, right, ploty, out = search_around_poly(warp(binary), left_coef, right_coef)
    ax2.clear()
    ax2.imshow(out)
    # ax2.plot(left, ploty, 'b')
    # ax2.plot(right, ploty, 'r')
    ax1.imshow(binary)
    f.canvas.draw_idle()

def change_picture(d):
    global cap
    global image
    ret, im = cap.read()
    if ret:
        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    upfate(0)
    print("yololo")


# Run the function first


# Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#
# axcolor = 'lightgoldenrodyellow'
# sl_min = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
# tresh_min = Slider(sl_min, "Thresh_min", 0, 255, valinit=43, valstep=1)
# sl_max = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
# tresh_max = Slider(sl_max, "Thresh_max", 0, 255, valinit=255, valstep=1)
#
# sl_min_ang = plt.axes([0.25, 0.8, 0.65, 0.03], facecolor=axcolor)
# tresh_min_ang = Slider(sl_min_ang, "Ang_min", 0, 255, valinit=43, valstep=1)
# sl_max_ang = plt.axes([0.25, 0.85, 0.65, 0.03], facecolor=axcolor)
# tresh_max_ang = Slider(sl_max_ang, "Ang_max", 0, 255, valinit=255, valstep=1)
#
#
# sl_min_col = plt.axes([0.25, 0.9, 0.65, 0.03], facecolor=axcolor)
# tresh_min_color = Slider(sl_min_col, "Color_min", 0, 255, valinit=31, valstep=1)
# sl_max_col = plt.axes([0.25, 0.95, 0.65, 0.03], facecolor=axcolor)
# tresh_max_color = Slider(sl_max_col, "Color_max", 0, 255, valinit=255, valstep=1)
#
# axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
# bcut = Button(axcut, 'NEXT', color='red', hovercolor='green')
# bcut.on_clicked(change_picture)
#
#
# tresh_min_ang.on_changed(upfate)
# tresh_max_ang.on_changed(upfate)
# tresh_max_color.on_changed(upfate)
# tresh_min_color.on_changed(upfate)
# tresh_min.on_changed(upfate)
# tresh_max.on_changed(upfate)
# f.tight_layout()
#
# binary = color_grad_binary(image,sobel_kernel=3, th=(tresh_min.val,tresh_max.val),
#                                    th_col=(tresh_min_color.val, tresh_max_color.val),
#                                    th_ang=(tresh_min_ang.val, tresh_max_ang.val))
# left_coef, right_coef, left, right, ploty, out = window_fit_polynomial(warp(binary))
#
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(out, cmap='gray')
# # ax2.plot(left, ploty, 'b')
# # ax2.plot(right, ploty, 'r')
# ax2.set_title('Thresholded Gradient', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()

def run_opencv(cap):


    is_first = True
    while True:
        is_ok, image = cap.read()

        if not is_ok:
            break

        image = cv2.undistort(image, mtx, dist, None, mtx)

        # harder
        binary = color_grad_binary(image, sobel_kernel=9,
                                          th=(15, 255),
                                          th_col=(31, 255),
                                          th_ang=(43, 255))

        # challenge
        # binary = color_grad_binary(image, sobel_kernel=9,
        #                            th=(15, 255),
        #                            th_col=(31, 255),
        #                            th_ang=(43, 255))

        if is_first:
            left_coef, right_coef, left, right, ploty, out, left_coef_m, right_coef_m = window_fit_polynomial(warp(binary))
            is_first = False
        else:
            left_coef, right_coef, left, right, ploty, out, left_coef_m, right_coef_m = search_around_poly(warp(binary), left_coef, right_coef)

        lpts = np.array(list(zip(left, ploty)), dtype=np.int)
        lpts = lpts[lpts[:, 1] >= 0]

        rpts = np.array(list(zip(right, ploty)), dtype=np.int)
        rpts = rpts[rpts[:, 1] >= 0]

        # print(lpts)
        cv2.polylines(out,[lpts], False, color=(0, 255, 255))
        cv2.polylines(out, [rpts], False, color=(0, 255, 255))
        cv2.fillPoly(out, [np.concatenate((lpts, rpts[::-1]))], color=(0, 255, 0))

        out_inv = warp(out, inverse=True)
        plotted = cv2.addWeighted(image, 1, out_inv, 0.3, 0)

        cv2.putText(image, str(measure_curvature_real(ploty, left_coef_m, right_coef_m)), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        im2show = np.concatenate((image, cv2.cvtColor(binary*255, cv2.COLOR_GRAY2BGR)), axis=1)
        im2show = np.concatenate((im2show, np.concatenate((out, plotted), axis=1)), axis=0)
        cv2.imshow("Result", im2show)
        cv2.waitKey(0)

run_opencv(cap)
