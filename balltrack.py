from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
#brownLower = (29, 86, 6)
#brownUpper = (64, 255, 255)

brownLower = (47,25,35)
brownUpper = (175,156,153)

farend = (118, 25, 105) # We may want to change this V value
lowend = (130, 52, 150)


fgbg2 = cv2.createBackgroundSubtractorMOG2();

rate = 0
pts = deque(maxlen=args["buffer"])

iters = 1

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
        
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])
    
#Creating a Pandas DataFrame To Store Data Point
Data_Features = ['x', 'y', 'time']
Data_Points = pd.DataFrame(data = None, columns = Data_Features , dtype = float)


#Reading the time in the begining of the video.
start = time.time()

# keep looping
while True:
	# grab the current frame
    (grabbed, frame) = camera.read()

	#Reading The Current Time
    current_time = time.time() - start

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

	# resize the frame, blur it, and convert it to the HSV
	# color space
    frame = imutils.resize(frame, width=500)
    background = fgbg2.apply(frame)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, farend, lowend)
    mask = cv2.erode(mask, None, iterations=iters)
    mask = cv2.dilate(mask, None, iterations=iters)

    mask = cv2.bitwise_and(background, mask, mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None


    if len(cnts) > 0:
		# Need to fix ripped code.

        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if (M["m00"] != 0):
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


        if  (radius < 10 ) :
            cv2.circle(frame, (int(x), int(y)), int(radius),
	           (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            #Save The Data
            Data_Points.loc[Data_Points.size/3] = [x , y, current_time]

	# update the points queue
    pts.appendleft(center)
    
	# loop over the set of tracked points
    for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
        if pts[i - 1] is None or pts[i] is None:
            continue
            
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        
	# show the frame to our screen
    framecopy = frame.copy()
    for cunt in cnts:
        cuntarea = cv2.contourArea(cunt)
        if (cuntarea > 75 and cuntarea < 1000):
            cv2.drawContours(framecopy, cunt, -1, (0, 255, 0))

    cv2.imshow("Contours", framecopy)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(rate) & 0xFF
        
	# if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    elif key == ord("s"):
        rate += 10
    elif key == ord("f"):
        rate -= 10
        if rate < 1 : rate = 1


        

#Ripped Data Collection

#'h' is the focal length of the camera
#'X0' is the correction term of shifting of x-axis
#'Y0' is the correction term ofshifting of y-axis
#'time0' is the correction term for correction of starting of time
h = 0.2
X0 = 0
Y0 = 0
time0 = 0
theta0 = 0.3

#Applying the correction terms to obtain actual experimental data
Data_Points['x'] = Data_Points['x']- X0
Data_Points['y'] = Data_Points['y'] - Y0
Data_Points['time'] = Data_Points['time'] - time0

#Calulataion of theta value
Data_Points['theta'] = 2 * np.arctan(Data_Points['y']*0.0000762/h)#the factor correspons to pixel length in real life
Data_Points['theta'] = Data_Points['theta'] - theta0

#Creating the 'Theta' vs 'Time' plot
plt.plot(Data_Points['theta'], Data_Points['time'])
plt.xlabel('Theta')
plt.ylabel('Time')

#Export The Data Points As cvs File and plot
Data_Points.to_csv('Data_Set.csv', sep=",")
plt.savefig('Time_vs_Theta_Graph.svg', transparent= True)

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()