from line_polinom import *
from polinom2 import *

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    left_curverad = np.power(1 + (2 * left_fit_cr[0] * y_eval * config.ym_per_pix + left_fit_cr[1]) ** 2, 1.5) / np.abs(
        2 * left_fit_cr[0])
    right_curverad = np.power(1 + (2 * right_fit_cr[0] * y_eval * config.ym_per_pix + right_fit_cr[1]) ** 2, 1.5) / np.abs(
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
        # empirically found parameters using project video
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


def color_grad_binary(img, th=(0, 255), th_grad=(-np.pi, np.pi), mag=False, plot=False):

    ### Convert color
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    h_channel = hls[:,:,0]
    l_channel = hls[:, :,1]
    s_channel = hls[:, :,2]

    s_channel = normalize(s_channel)
    ## Good way to find yellow line in s_channel
    mask1 = sobel_x_binary(s_channel, th=th, ksize=11)

    ## Find white lines by applying gradient technique
    if mag:
        mask2 = mag_thresh_binary(l_channel, mag_thresh=th)
    else:
        mask2 = sobel_x_binary(l_channel, th=th_grad, ksize=3)


    ## In challenge_video there are black line with good gradients, i want to remove it
    mask3 = filter_color(l_channel, th=(np.mean(l_channel), 255))
    mask3 = cv2.GaussianBlur(mask3*255, ksize=(3,3), sigmaX=1)
    mask3[mask3 < 255] = 0
    mask3 = np.uint8(mask3/255.)

    ## Finilize results of gradien/color thresholds
    mask = np.zeros_like(mask1)
    mask[((mask2 == 1) | (mask1 == 1)) & (mask3 == 1)] = 1

    if plot:
        cv2.imshow("mask1", mask1 * 255)
        cv2.imshow("mask2", mask2 * 255)
        cv2.imshow("mask3", mask3 * 255)

    return  mask

def find_shift(imshape, left_coef, right_coef):
    """
    Find out car shift from lane center (meters)
    """
    # Calculate vehicle center offset in pixels
    ymax = imshape[0] - 1
    bottom_x_left  = left_coef[0] * (ymax**2) + left_coef[1] * ymax + left_coef[2]
    bottom_x_right = right_coef[0] * (ymax**2) + right_coef[1] * ymax + right_coef[2]
    car_shift = imshape[1]/2 - (bottom_x_left + bottom_x_right)/2

    # Convert pixel offset to meters
    car_shift *= config.xm_per_pix

    return car_shift