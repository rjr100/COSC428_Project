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

#create tracker object
tracker = EuclideanDistTracker()

backSub1 = cv.createBackgroundSubtractorMOG2(history=10, varThreshold=20,detectShadows=False)
files = ['./Footage/COSC_Footage.mp4','./Footage/players2.mp4','./Footage/one_player.mp4','./Footage/COSC_Indoor_Dim.mp4']
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
        
        for pos in ball_pos_buf:
            errors_sq.append((a_opt*(pos[0]-b_opt)**2 + c_opt - pos[1])**2) 
        ax.set_xlim(0,640)
        ax.set_ylim(0,480)
        ax.scatter(X,y)
        ax.plot(x_model,y_model, color='r')
        plt.show()
    except:
        print("Parameters not found")
        return

    while(1):
        keyboard = cv.waitKey()
        if keyboard & 0xFF == ord('w'):
            plt.close()
            return
    #except:
        #print('Couldnt fit a curve')

def create_mask(frame):
    start_time = time.time()
    fgMask1 = backSub1.apply(frame)
    
    median = cv.medianBlur(fgMask1, 3)     # doing this twice does a better job of removing salt & pepper noise
    #median2 = cv.medianBlur(median, 5)     # doing this twice does a better job of removing salt & pepper noise
    ##opened = cv.morphologyEx(fgMask1, cv.MORPH_OPEN, np.ones((5, 5)))
    closed = cv.morphologyEx(median, cv.MORPH_CLOSE, np.ones((15, 15)))
    finish_time = time.time()
    time_taken = finish_time - start_time
    #print('mask time = ' + str(time_taken))	
    #closed = fgMask1
    cv.imshow('MASK', fgMask1)
    cv.imshow('Morphology', closed)  
    return closed, time_taken

def find_ball_ellipse(frame, mask, distortion):
    height, width, _ = frame.shape
    
    #fgMask1 = backSub1.apply(frame)
    
    #median = cv.medianBlur(fgMask1, 9)     # doing this twice does a better job of removing salt & pepper noise
    #median2 = cv.medianBlur(median, 9)     # doing this twice does a better job of removing salt & pepper noise
    ##opened = cv.morphologyEx(fgMask1, cv.MORPH_OPEN, np.ones((5, 5)))
    #closed = cv.morphologyEx(median2, cv.MORPH_CLOSE, np.ones((21, 21)))
    #cv.imshow('MASK', closed)
    #edges = cv.Canny(closed,100,200)
    #result = hough_ellipse(edges, threshold=250,
                       #min_size=100, max_size=120)
    #cv.imshow('Result', result)
    #cv.imshow('CANNY', edges)
    
    start_time = time.time()
    # Object detection
    contours, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    end_time = time.time()
    #cv.imshow('Contours', contours)
    detections = []
    ellipse = []
    position = []
    ball_ellipse = None
    is_ball = False    
    
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
               
                CD = area/ellipse_A
                ratio = ellipse_w/ellipse_h
                #print(ratio)
                if CD > 0.95:
                    pass_CD = True
                else:
                    pass_CD = False
                if 0.4 < ratio <= 1:
                    pass_ratio = True
                else:
                    pass_ratio = False  
                #print("Pass CD: " + str(pass_CD) + " Pass ratio:" + str(pass_ratio))
                if pass_CD and pass_ratio:
                    is_ball = True
                    cv.ellipse(frame, ellipse, (0, 255,0), 3) 
                    cv.putText(frame, 'Ball', [int(centerX)+10,int(centerY)+10], 0, 1, (0,0,255),2)
                    ball_ellipse = ellipse
                    position = [centerX, centerY]
                    
                    return detections, position, ball_ellipse, is_ball
                else:
                    is_ball = False        
                    #cv.putText(frame, 'Player', [int(centerX)+10,int(centerY)+10], 0, 1, (0,255,255),2)
                    ball_ellipse = None
                    position = []
                #cv.putText(frame, str(CD), [int(centerX)+10,int(centerY)+10], 0, 1, (0,0,255),2)
            i += 1
            if i > 4:
                break
    return detections, position, ball_ellipse, is_ball

def track_ball(detections, frame):
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(frame, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)    

def main(file=-1):
    cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    
    capture = cv.VideoCapture(file)
    
    w = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    print((w, h))
    
    if (file != -1):
        distortion = (w/640, h/480)
        cv.resizeWindow("Frame", 640, int(h/(w/640)))
    #fps = capture.get(cv.CAP_PROP_FPS)
    fps = 1/wait_time
    dt = 1/fps
    ballpos = []
    images = []
    tracked_points = []
    ball_count = 0
    no_ball = 0
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
        mask, mask_time = create_mask(frame)
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
        
        # If you get 5 successful ball readings, fit a parabola
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
