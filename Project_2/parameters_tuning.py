from algorithms import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

############################################################
                ##VIZUALIZER#
############################################################

cap = cv2.VideoCapture("project_video.mp4")

re, image = cap.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def upfate(val):
    global left_coef
    global right_coef
    binary = color_grad_binary(image, th=(tresh_min.val,tresh_max.val),
                               th_grad=(tresh_min_ang.val, tresh_max_ang.val))
    left_coef, right_coef, left, right, ploty, out, _, _ = search_around_poly(warp(binary), left_coef, right_coef)
    ax2.clear()
    ax2.imshow(out)
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
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

axcolor = 'lightgoldenrodyellow'
sl_min = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
tresh_min = Slider(sl_min, "Thresh_min", 0, 255, valinit=43, valstep=1)
sl_max = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
tresh_max = Slider(sl_max, "Thresh_max", 0, 255, valinit=255, valstep=1)

sl_min_ang = plt.axes([0.25, 0.8, 0.65, 0.03], facecolor=axcolor)
tresh_min_ang = Slider(sl_min_ang, "Ang_min", 0, 255, valinit=43, valstep=1)
sl_max_ang = plt.axes([0.25, 0.85, 0.65, 0.03], facecolor=axcolor)
tresh_max_ang = Slider(sl_max_ang, "Ang_max", 0, 255, valinit=255, valstep=1)

axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
bcut = Button(axcut, 'NEXT', color='red', hovercolor='green')
bcut.on_clicked(change_picture)


tresh_min_ang.on_changed(upfate)
tresh_max_ang.on_changed(upfate)
tresh_min.on_changed(upfate)
tresh_max.on_changed(upfate)
f.tight_layout()

binary = color_grad_binary(image, th=(tresh_min.val,tresh_max.val),
                           th_grad=(tresh_min_ang.val, tresh_max_ang.val))
left_coef, right_coef, left, right, ploty, out, _, _ = window_fit_polynomial(warp(binary))

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(out, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()