from __future__ import print_function
import cv2 as cv
import time
import argparse
from tracker import *
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
"""
Questions for the homies
1. How can I only use the maximum found contour
2. Fix my Kalman filter pls
3. Display Kalman filter stuff
4. Display tragectory
5. How to tell if the shape is elliptical? Can I iterate through the shape on the mask and see if
it is mostly white?


Compare 
- Using median filter vs using open/close
- Using ellipse vs using Centroids
- Using Kalman filtering vs not to see if the graph is more accurate



Report sections:
- Mask
- Detection
- Evaluation metrics
"""

wait_time = 10

#parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              #OpenCV. You can process both videos and images.')
#parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default=-1)
#parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
#args = parser.parse_args()

#create tracker object
tracker = EuclideanDistTracker()

backSub1 = cv.createBackgroundSubtractorMOG2(history=10, varThreshold=5,detectShadows=False)
files = ['./Footage/lineout1.mp4','./Footage/players.mp4','./Footage/test4.mp4', './Footage/test2.mp4','./Footage/test4.mp4', './Footage/test6.mp4', -1]



def find_ball_ellipse(frame, distortion):
    height, width, _ = frame.shape
    
    fgMask1 = backSub1.apply(frame)
    
    #median = cv.medianBlur(fgMask1, 9)     # doing this twice does a better job of removing salt & pepper noise
    #median2 = cv.medianBlur(median, 9)     # doing this twice does a better job of removing salt & pepper noise
    opened = cv.morphologyEx(fgMask1, cv.MORPH_OPEN, np.ones((5, 5)))
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, np.ones((21, 21)))
    #edges = cv.Canny(closed,100,200)
    #result = hough_ellipse(edges, threshold=250,
                       #min_size=100, max_size=120)
    #cv.imshow('Result', result)
    #cv.imshow('CANNY', edges)
    
    # Object detection
    contours, _ = cv.findContours(closed, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    #cv.imshow('Contours', contours)
    detections = []
    ellipse = []
    position = []
    ball_ellipse = None
    is_ball = False    
    
    
    #for cnt in contours:
        #cv.drawContours(frame, cnt, 0, (255,255,0), 5)
    if len(contours) > 0:
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        i = 0
        #cnt = max(contours, key=cv.contourArea)  
        for cnt in sorted_contours:
            area = cv.contourArea(cnt)
            #cv.drawContours(frame, cnt, 0, (255,255,0), 5)
            if area > 200:
                ellipse = cv.fitEllipse(cnt)
                centerX = ellipse[0][0]
                centerY = ellipse[0][1]
                ellipse_w = ellipse[1][0]
                ellipse_h = ellipse[1][1]
                pi = round(np.pi, 4)
                ellipse_A = pi*(ellipse_w/2)*(ellipse_h/2)
               
                CD = area/ellipse_A
                ratio = ellipse_w/ellipse_h
                print(ratio)
                if CD > 0.95:
                    pass_CD = True
                else:
                    pass_CD = False
                if 0.4 < ratio <= 1:
                    pass_ratio = True
                else:
                    pass_ratio = False  
                    
                if pass_CD and pass_ratio:
                    is_ball = True
                    cv.ellipse(frame, ellipse, (0, 255,0), 3) 
                    cv.putText(frame, 'Ball', [int(centerX)+10,int(centerY)+10], 0, 1, (0,0,255),2)
                    ball_ellipse = ellipse
                    position = [centerX, centerY]
                    return detections, position, ball_ellipse, is_ball
                else:
                    is_ball = False        
                    cv.putText(frame, 'Player', [int(centerX)+10,int(centerY)+10], 0, 1, (0,255,255),2)
                    ball_ellipse = None
                    position = []
                #cv.putText(frame, str(CD), [int(centerX)+10,int(centerY)+10], 0, 1, (0,0,255),2)
                #print(ellipse)
                #cv.ellipse(closed, ellipse, (0, 255,0), 3)
                #cv.circle(frame, (int(centerX), int(centerY)), 5, (0,255,0), 5)    
                
            i += 1
            if i > 4:
                break
    

    cv.imshow('MASK', closed)
    
    return detections, position, ball_ellipse, is_ball

def track_ball(detections, frame):
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(frame, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)    


def fx(x, dt):
    # state transition function - predict next state based
    # on constant velocity model x = vt + x_0
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]], dtype=float)
    return np.dot(F, x)

def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]
    return np.array([x[0], x[2]])

def main(file=-1):
    cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    
    capture = cv.VideoCapture(file)
    
    w = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    print((w, h))
    distortion = (w/640, h/480)
    cv.resizeWindow("Frame", 640, 480) 
    time.sleep(1)
    #fps = capture.get(cv.CAP_PROP_FPS)
    fps = 1/wait_time
    dt = 1/fps
    ballpos = []

    while capture.isOpened():
        ret, in_frame = capture.read()
        

        if in_frame is None:
            break
        
        frame = cv.resize(in_frame,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
        # Ball detecting
        detections, position, ellipse, is_ball = find_ball_ellipse(frame, distortion)
        #print(ellipse)
        # Object tracking
        #track_ball(detections, frame)
        #Kalman filtering
        
        ##points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
        ##kalman = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)
        ##kalman.x = np.array([1,1,1,1])  # Initial State
        ##kalman.P = np.array([[2,0,0,0],
                             ##[0,2,0,0],
                             ##[0,0,2,0],
                             ##[0,0,0,2]], np.float32)  # Covariance Matrix
        ##kalman.R = np.array([[1,0],
                             ##[0,1]], np.float32)  # Measurement Noise
        ##kalman.Q = np.array([[1,0,0,0],
                             ##[0,1,0,0],
                             ##[0,0,100,0],
                             ##[0,0,0,100]], np.float32)  # Process Noise=2)
        
        ##kalman.predict()  # Predict the ball's position.
        ##center_predict = (int(kalman.x[0]), int(kalman.x[2]))
        
        #if len(position) > 0:
            ## The Kalman filter expects the x,y coordinates in a 2D array.
            #measured = np.array([position[0], position[1]], dtype="float32")
            ## Update the Kalman filter with the current ball location if we have it.
            #kalman.update(measured)

   
        ## Draw an ellipse showing the uncertainty of the predicted position.
        #center = (int(kalman.x[0]), int(kalman.x[2]))
        ##print(center[0] - center_predict[0], center[1] - center_predict[1])
        ##print(int(kalman.x[0]), int(kalman.x[1]))
        #axis_lengths = (int(kalman.P_prior[0, 0]), int(kalman.P_prior[1, 1]))
        
        
        ##print(position)
        if(position != []):
            if len(ballpos) < 10:
                if position[0] > 0 and position[1] > 0:
                    ballpos.append(position)
                    #print(ballpos)
            else:
                if position[0] > 0 and position[1] > 0:
                    ballpos.pop(0)
                    ballpos.append(position)
                    #print(ballpos)                
                    
        if len(ballpos) > 1:
            for pos in ballpos:
                x = pos[0]
                y = pos[1]
                cv.circle(frame, (int(x),int(y)), 2, (0,255,0), 5)
        
        #if ellipse != []:
            #cv.ellipse(frame, (center, ellipse[1], ellipse[2]), (255, 0, 0), 3) #blue
            ##cv.ellipse(frame, (center_predict, ellipse[1], ellipse[2]), (0, 0, 255), 3) #red
        
        cv.imshow('Frame', frame)
        #cv.imshow('MOG2', median2)   
        keyboard = cv.waitKey()
        if keyboard & 0xFF == ord('q'):
            break
    capture.release()

if __name__ == "__main__":
    #main()
    for file in files:
        main(file)
    cv.destroyAllWindows()       