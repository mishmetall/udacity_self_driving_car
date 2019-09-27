from algorithms import *
from camera_calibrate import calibrate_camera

from tqdm import tqdm



def process_video(path, out_path, plot=False):
    """ Process videofile """

    cap = cv2.VideoCapture(path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_cap = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))

    # Calibrate camera or load existing calibration
    dist, mtx = calibrate_camera("camera_cal")

    is_first = True
    for i in tqdm(range(length)):
        is_ok, image = cap.read()

        if not is_ok:
            break

        # 1. Undistort image
        image = cv2.undistort(image, mtx, dist, None, mtx)

        # 2. Filter lanes on the video
        binary = color_grad_binary(image, th=(15, 255),
                                   th_grad=(23, 42))

        # 3. If image is first in sequence, apply histogram-based polynomial fitting, else, refine using previous estimate
        if is_first:
            left_coef, right_coef, left, right, ploty, out, left_coef_m, right_coef_m = \
                window_fit_polynomial(warp(binary))
            is_first = False
        else:
            left_coef, right_coef, left, right, ploty, out, left_coef_m, right_coef_m = \
                search_around_poly(warp(binary), left_coef, right_coef)
        
        lpts = np.array(list(zip(left, ploty)), dtype=np.int)
        lpts = lpts[lpts[:, 1] >= 0]

        rpts = np.array(list(zip(right, ploty)), dtype=np.int)
        rpts = rpts[rpts[:, 1] >= 0]

        cv2.polylines(out,[lpts], False, color=(0, 255, 255))
        cv2.polylines(out, [rpts], False, color=(0, 255, 255))
        
        cv2.fillPoly(out, [np.concatenate((lpts, rpts[::-1]))], color=(0, 255, 0))

        out_inv = warp(out, inverse=True)
        plotted = cv2.addWeighted(image, 1, out_inv, 0.3, 0)

        curvature = measure_curvature_real(ploty, left_coef_m, right_coef_m)
        curvature = (curvature[0] + curvature[1]) / 2

        shift = find_shift(image.shape, left_coef, right_coef)
        cv2.putText(plotted, "Radius: %3.2f m" % (curvature,), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(plotted, "Shift: %3.2f m" % (shift,), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if plot:
            im2show = np.concatenate((image, cv2.cvtColor(binary*255, cv2.COLOR_GRAY2BGR)), axis=1)
            im2show = np.concatenate((im2show, np.concatenate((out, plotted), axis=1)), axis=0)
            cv2.imshow("Result", im2show)
            cv2.waitKey(0)

        out_cap.write(plotted)

    cap.release()
    out_cap.release()


if __name__=="__main__":

    process_video("challenge_video.mp4", "challenge_video_result.mp4")
    process_video("project_video.mp4", "project_video_result.mp4")
