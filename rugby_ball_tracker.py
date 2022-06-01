from __future__ import print_function
import cv2 as cv
import time
import argparse
from tracker import *
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from scipy.optimize import curve_fit

# MACROS
wait_time = 10

# Set up Mixture of Gaussian 2 BS
backSub1 = cv.createBackgroundSubtractorMOG2(history=10, varThreshold=20,detectShadows=False)

files = ['./Footage/COSC_Footage.mp4','./Footage/players2.mp4','./Footage/COSC_Indoor_Dim.mp4']
#files = ['./Footage/players2.mp4','./Footage/one_player.mp4']

def model_f(x, a, b, c):
    return a*(x-b)**2 + c

def average(array):
    return sum(array)/len(array)

def fit_parabola(ball_pos, image, h):
    try:
        ball_pos_buf = ball_pos
        for pos in ball_pos:
            pos[1] = h - pos[1]
        dt = np.array(ball_pos)
        
        # Preparing X and y data from the given data
        X = dt[:, 0]
        y = dt[:, 1]
        
        # Calculating parameters (Here, intercept-theta1 and slope-theta0)
        # of the line using the numpy.polyfit() function
        popt, pcov = curve_fit(model_f, X, y, p0=[1,1,1])
        fig, ax = plt.subplots()
        
        a_opt, b_opt, c_opt = popt
        x_model = np.linspace(min(X), max(X), 100)
        y_model = model_f(x_model, a_opt, b_opt, c_opt)
        errors_sq = []
        
        # Get accuracy metric
        for pos in ball_pos_buf:
            errors_sq.append((a_opt*(pos[0]-b_opt)**2 + c_opt - pos[1])**2)
        sum_errors = sum(errors_sq)
        av_error = sum(errors_sq)/len(errors_sq)
        
        print(av_error)
        ax.set_xlim(0,640)
        ax.set_ylim(0,480)
        ax.scatter(X,y)
        ax.plot(x_model,y_model, color='r')
        plt.show() 

    # Convergence depends on initial guess which isn't always accurate
    except:
        print("Parameters not found")
        return

    while(1):
        keyboard = cv.waitKey()
        if keyboard & 0xFF == ord('w'):
            plt.close()
            return

# Create foreground mask
def create_mask(frame):
    start_time = time.time()
    fgMask1 = backSub1.apply(frame)
    
    median = cv.medianBlur(fgMask1, 3)     # Removing salt & pepper noise
    closed = cv.morphologyEx(median, cv.MORPH_CLOSE, np.ones((15, 15))) # Close
    
    cv.imshow('MASK', fgMask1)
    cv.imshow('Morphology', closed)  
    return closed

def find_ball_ellipse(frame, mask, distortion):
    height, width, _ = frame.shape
    
    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    
    detections = []
    ellipse = []
    position = []
    ball_ellipse = None
    is_ball = False    
    
    # Filter contours
    start_time1 = time.time()
    if len(contours) > 0:
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        i = 0
        for cnt in sorted_contours:
            area = cv.contourArea(cnt)
            if 2000 > area > 100:
                ellipse = cv.fitEllipse(cnt)
                centerX = ellipse[0][0]
                centerY = ellipse[0][1]
                ellipse_w = ellipse[1][0]
                ellipse_h = ellipse[1][1]
                pi = round(np.pi, 4)
                ellipse_A = pi*(ellipse_w/2)*(ellipse_h/2)

                # Compactness degree
                CD = area/ellipse_A
                if CD > 0.95:
                    pass_CD = True
                else:
                    pass_CD = False

                # Aspect ratio
                ratio = ellipse_w/ellipse_h
                if 0.4 < ratio <= 1:
                    pass_ratio = True
                else:
                    pass_ratio = False  
                
                # Only label as a ball if it passes all filters
                if pass_CD and pass_ratio:
                    is_ball = True
                    cv.ellipse(frame, ellipse, (0, 255,0), 3) 
                    cv.putText(frame, 'Ball', [int(centerX)+10,int(centerY)+10], 0, 1, (0,0,255),2)
                    ball_ellipse = ellipse
                    position = [centerX, centerY]
                    return detections, position, ball_ellipse, is_ball
                # Otherwise label as player
                else:
                    is_ball = False        
                    #cv.putText(frame, 'Player', [int(centerX)+10,int(centerY)+10], 0, 1, (0,255,255),2)
                    ball_ellipse = None
                    position = []
            i += 1
            if i > 4:
                break
    return detections, position, ball_ellipse, is_ball

def main(file=-1):
    cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    
    capture = cv.VideoCapture(file)
    
    w = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    print((w, h))
    
    # Scale image down to 640 on the x axis
    # Keep aspect ratio the same so the elliptical shape doesnt get affected
    # Tune error for different input resolutions default (1920, 1080)
    if (file != -1):
        distortion = (w/640, h/480)
        cv.resizeWindow("Frame", 640, int(h/(w/640)))
    
    fps = 1/wait_time
    dt = 1/fps
    ballpos = []
    images = []
    tracked_points = []
    ball_count = 0
    no_ball = 0
    
    # To save footage, set to True
    want_output = False
    if want_output:
        out_file = './Out_footage/output_footage' + str(time.ctime()) +'.avi'
        out = cv.VideoWriter(out_file,cv.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
    frame_index = 0
    frame_saved = False
    
    while capture.isOpened():
        #print(frame_index)
        ret, in_frame = capture.read()
        if in_frame is None:
            break
        new_h = int(h/(w/640))
        frame = cv.resize(in_frame,(640,new_h),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
        
        # Create Mask
        mask = create_mask(frame)
        
        # Ball detecting
        detections, position, ellipse, is_ball = find_ball_ellipse(frame, mask, distortion)
  
        if(position != []):
            if len(ballpos) < 15:
                if position[0] > 0 and position[1] > 0:
                    ballpos.append(position)
                    #print(ballpos)
            else:
                if position[0] > 0 and position[1] > 0:
                    ballpos.pop(0)
                    ballpos.append(position)
                    #print(ballpos)        

        # Mark Ball locations with green circles
        if len(ballpos) > 1:
            for pos in ballpos:
                x = pos[0]
                y = pos[1]
                cv.circle(frame, (int(x),int(y)), 1, (0,255,0), 3)
        
        # If you get enough successful ball readings, fit a parabola
        if is_ball == True:
            ball_count += 1
            no_ball = 0
            #print(ball_count)
            if ball_count >= 6 and not frame_saved:
                #images.append(frame)
                #tracked_points.append(ballpos)
                frame_saved = True
        else:
            no_ball += 1
            if no_ball > 2:
                if frame_saved:
                    #print(ballpos)
                    frame_saved = False
                    fit_parabola(ballpos, frame, new_h)                    
                ball_count = 0
                ballpos = []
        # Write to output if enabled
        if want_output:
            out.write(mask)
        cv.imshow('Frame', frame)
        frame_index+=1       
        
        keyboard = cv.waitKey()
        if keyboard & 0xFF == ord('q'):
            break
    
    capture.release()
    if want_output:
        out.release()   

if __name__ == "__main__":
    #main()
    for file in files:
        main(file)
    cv.destroyAllWindows()       
