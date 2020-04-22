from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep


def circleDif(p):
    circ = cv2.minEnclosingCircle(p)

    return abs((circ[1]**2 * np.pi) - cv2.contourArea(p)) if circ[1] > 5 else float("Inf")

# construct the argument parse and parse the arguments
# USAGE: Add the -v tag and then the video name (Shots.mp4)
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# brownLower = (47,25,35)
# brownUpper = (175,156,153)

# These are largely tuning variables. The current pair in use are farend and lowend. These are HSV ranges for acceptable colors
brownLower = (47,25,35)
brownUpper = (175,156,153)

farend = (118, 25, 105) # We may want to change this V value
lowend = (130, 52, 150)

#Denotes where the center of the hoop is and the ball width. This is used for basic prediction
YTRUTH = 60
XTRUTH = 198
BALL_WIDTH = 22.5

# Background subtraction
fgbg2 = cv2.createBackgroundSubtractorMOG2();

# Framerate (ms)
rate = 0

#Accuracy variables
accuracy = 0
olderror = 5

# ?

pts = deque(maxlen=args["buffer"])
iters = 1

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    print('Must use Shots.mp4... Try again')
    quit()
    #camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# Initialize found points vector to meaningless value (will be instantly reset)
points = [(0, 0)]

#Step up so the pyplot is non-blocking
plt.axis([150,400,-150,300])

plt.ion()
plt.show()




# keep looping
while True:
	# grab the current frame
    (grabbed, frame) = camera.read()


	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

	# resize the frame, blur it, and convert it to the HSV
	# color space
    frame = imutils.resize(frame, width=500)

    #Apply background subtraction to the current frame to obtain the filter for forground objects
    background = fgbg2.apply(frame)

    # Obtain a filter based on color of objects
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, farend, lowend)
    mask = cv2.erode(mask, None, iterations=iters)
    mask = cv2.dilate(mask, None, iterations=iters)

    # Create a mask that accounts for color and foreground objects.
    mask = cv2.bitwise_and(background, mask, mask)

    # Find contours based on the color/movement mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:

        #Contours section

        # Pick the largest contour that was found, place a circle around it
        c = min(cnts, key=circleDif)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if (M["m00"] != 0):
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Choose to either add to a list of points or reset the list of points
            if (center[0] > points[-1][0] + 25):
                points = [center]
                plt.clf()
                print('reset\n')
            else:
                points.append(center)

            # Given enough points lets try to make some predictions.
            if (len(points) > 5):

                #Area for prediction and pyplot stuff

                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]
                fit, error, _, _, _ = np.polyfit(x_coords, y_coords, 2, full=True)
                f = np.poly1d(fit)

                #Accuracy check
                ycheck = f(XTRUTH)
                accuracy = 1 - abs(YTRUTH - ycheck)/BALL_WIDTH
                accuracy = 0 if accuracy < 0 else accuracy * 100

                # if (2 * olderror < error/len(points)):
                #     accuracy *= olderror / (error/len(points))

                olderror = error/len(points)
                #print(' x: {} \n y: {}'.format(x_coords,y_coords))
                #print(x,y)
                print("Chance to make it: {:.4F}".format(accuracy))

                xp = np.linspace(150, 400, 200)
                plt.ylim(-250,500)
                plt.scatter(XTRUTH,YTRUTH,label = 'Hoop Location', c = 'blue', marker = 'x')
                plt.scatter(x, y, label = 'Ball Postition')
                plt.scatter(x_coords,y_coords, label = 'Tracked Postitions')
                plt.plot(xp, f(xp), '--', label = 'Trajectory')

                plt.title('Tracked Ball postions and parabolic regression lines \n (x,y) are relative to top left of image')
                plt.xlabel('x pixel location')
                plt.ylabel('y pixel location')
                #plt.legend()

                plt.draw()
                plt.pause(10**-4)


        if  (radius < 10 ) :
            #Draws circle on frame (yellow)
            cv2.circle(frame, (int(x), int(y)), int(radius),
	           (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


	# update the points queue
    pts.appendleft(center)

    framecopy = frame.copy()
	# loop over the set of tracked points
    for i in range(1, len(pts)):
        #adds red trail on trame

		# if either of the tracked points are None, ignore
		# them
        if pts[i - 1] is None or pts[i] is None:
            continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
    for c in cnts:
        c_area = cv2.contourArea(c)
        if (c_area > 75 and c_area < 1000):
            cv2.drawContours(framecopy, c, -1, (0, 255, 0))

    #accuracy print statement on frame window
    cv2.putText(frame,str(accuracy)[0:5] + "%", (25,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("Contours", framecopy)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(rate) & 0xFF

	# if the 'q' key is pressed, stop the loop
    # 'j' for faster framerate processing
    # 'l' for slower ...
    if key == ord("q"):
        break
    elif key == ord("j"):
        rate += 10
    elif key == ord("l"):
        rate -= 10
        if rate < 1 : rate = 1


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
