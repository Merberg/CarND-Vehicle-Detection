
# coding: utf-8

# # Advanced Lane Finding  #
# This files calibrates images from a vehicles central camera for use in detecting lanes.
#
# ### Imports and Constants ###
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import operator
import random
from moviepy.editor import VideoFileClip
from IPython.display import HTML
get_ipython().magic('matplotlib inline')

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
LANE_LENGTH_m = 30
LANE_WIDTH_m = 3.7
LANE_WIDTH_pixel = 700
LANE_HALF_W_pixel = LANE_WIDTH_pixel/2
M_PER_PIXEL_y = LANE_LENGTH_m/IMAGE_HEIGHT
M_PER_PIXEL_x = LANE_WIDTH_m/LANE_WIDTH_pixel
BIRDS_EYE_OFFSET = np.absolute(LANE_WIDTH_pixel - IMAGE_WIDTH)/2
BIRDS_EYE_SRC = np.float32([(577, 460), (709, 460), (1018, 650), (BIRDS_EYE_OFFSET, 650)])
BIRDS_EYE_DST = np.float32([(BIRDS_EYE_OFFSET, 0), (IMAGE_WIDTH-BIRDS_EYE_OFFSET, 0),
                          (IMAGE_WIDTH-BIRDS_EYE_OFFSET, IMAGE_HEIGHT), (BIRDS_EYE_OFFSET, IMAGE_HEIGHT)])
N_LANE_WINDOWS = 8
LANE_WINDOW_HEIGHT = np.int(IMAGE_HEIGHT/N_LANE_WINDOWS)
LANE_WINDOW_WIDTH = 140

# ### Camera Calibration ###
# Prior to executing the lane finding pipeline, the camera must be calibrated using 9x6 chessboard images.  This is only completed once.
#
def findChessboardPoints(fname):
    # Find the chessboard corners of file.
    iRGB = mpimg.imread(fname)

    # Search for the image chessboard points
    gray = cv2.cvtColor(iRGB, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # Store the points if found
    if ret == True:
        worldPoints.append(worldPt)
        imagePoints.append(corners)
    return


# Process all the provided calibration images and run the calibration.
def calibrateCamera():
    nx = 9
    ny = 6
    worldPoints = []
    worldPt = np.zeros((nx*ny,3), np.float32)
    worldPt[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    imagePoints = []
    images = glob.glob('camera_cal/calibration*.jpg')
    for chessboard in images:
        findChessboardPoints(chessboard)

    print("Calibration points found in {} of {} images".format(len(imagePoints), len(images)))

    ret, cameraMtx, distCoeffs, rotVecs, transVecs = cv2.calibrateCamera(
        worldPoints, imagePoints, iOrig.shape[0:2], None, None)
    iUndistort = cv2.undistort(iOrig, cameraMtx, distCoeffs, None, cameraMtx)
    return

# ### Lane Finding Pipeline ###
def undistortImage(img):
    # Undistort the image by refining the camera matrix,
    # running the function, then cropping to the region of interest
    return cv2.undistort(img, cameraMtx, distCoeffs, None, cameraMtx)

def applyThreshold(binary, thresh):
    # Apply the threshold
    binary_th = np.zeros_like(binary)
    binary_th[(binary >= thresh[0]) & (binary <= thresh[1])] = 1
    return binary_th

def combineThresholdsForPlots(img):
    # Combine thresholds for testing with visualization
    # Convert to CIELAB color space and use the L channel for whites
    # and the b channel (blues to yellows) for yellows
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float)
    whiteB = applyThreshold(lab[:,:,0], (210,255))
    yellowB = applyThreshold(lab[:,:,2], (150,255))
    combined = np.zeros_like(whiteB)
    combined[((whiteB == 1)|(yellowB == 1))] = 1
    return whiteB, yellowB, combined

def combineThresholds(img):
    # Combine thresholds for pipeline use
    combine1, combine2, combined = combineThresholdsForPlots(img)
    return combined

def warpToBirdsEye(img):
    # Perform a perspective transform for the region of interest
    M = cv2.getPerspectiveTransform(BIRDS_EYE_SRC, BIRDS_EYE_DST)
    return cv2.warpPerspective(img, M, (IMAGE_WIDTH, IMAGE_HEIGHT))

def warpFromBirdsEye(img):
    Minv = cv2.getPerspectiveTransform(BIRDS_EYE_DST, BIRDS_EYE_SRC)
    return cv2.warpPerspective(img, Minv, (IMAGE_WIDTH, IMAGE_HEIGHT))

def findLaneStarts(img):
    # Calculate the histogram to use the peaks
    # as the start of the lanes
    histogram = np.sum(img[int(IMAGE_HEIGHT/2):,:], axis=0)
    center = np.int(histogram.shape[0]/2)
    xLeft = np.argmax(histogram[:center])
    xRight = center + np.argmax(histogram[center:])
    return xLeft, xRight

def findLaneInWindow(x, xDelta, y, nonzero):
    # Find the individual lane within the window
    y_nonzero = np.array(nonzero[0])
    x_nonzero = np.array(nonzero[1])

    # Locate one lane within the frame of a window
    windowHalfWidth = int(LANE_WINDOW_WIDTH/2)
    windowLeft = x - windowHalfWidth;
    windowRight = x + windowHalfWidth
    windowTop = y - LANE_WINDOW_HEIGHT
    windowBottom = y

    # Find the nonzero pixels in x and y within the window
    laneIdx = ((y_nonzero >= windowTop) &
               (y_nonzero < windowBottom) &
               (x_nonzero >= windowLeft) &
               (x_nonzero <= windowRight)).nonzero()[0]

    if len(laneIdx) > 0:
        xNext = np.int(np.mean(x_nonzero[laneIdx]))
        xDelta = (xDelta + (xNext-x))/2
    else:
        xNext = x + xDelta

    return xNext, xDelta, laneIdx, (windowLeft, windowBottom)

def findInitialLaneInImage(img, xStart, visualizeOn=False):
    # Find the individual lane within the image by
    # looking through windows
    nonzero = img.nonzero()
    y_nonzero = np.array(nonzero[0])
    x_nonzero = np.array(nonzero[1])
    laneIndices = []

    # Step through the windows one by one
    x = xStart
    xDelta = 0
    for w in range(N_LANE_WINDOWS+1):
        y = IMAGE_HEIGHT-(w)*LANE_WINDOW_HEIGHT
        x, xDelta, idx, windowCorner = findLaneInWindow(x, xDelta, y, nonzero)
        laneIndices.append(idx)

        # Plot rectangles if enabled
        if visualizeOn:
            windowFrame = patches.Rectangle(windowCorner,
                                            LANE_WINDOW_WIDTH,
                                            LANE_WINDOW_HEIGHT,
                                            fill=False,
                                            edgecolor='green')
            plt.gca().add_patch(windowFrame)

    # Flatten the indices to fit the polynomial
    laneIndices = np.concatenate(laneIndices)
    laneCoeff = np.polyfit(y_nonzero[laneIndices],
                           x_nonzero[laneIndices], 2)
    return laneCoeff, x_nonzero[laneIndices], y_nonzero[laneIndices]

def findLaneInImage(img, laneCoeff):
    # Find the individual lane within the image by either
    # looking in the windows or using the second order polynomial
    nonzero = img.nonzero()
    y_nonzero = np.array(nonzero[0])
    x_nonzero = np.array(nonzero[1])
    laneIndices = (
        (x_nonzero >
        (laneCoeff[0]*(y_nonzero**2) +
         laneCoeff[1]*y_nonzero + laneCoeff[2] - LANE_WINDOW_WIDTH))
        &
        (x_nonzero <
        (laneCoeff[0]*(y_nonzero**2) +
         laneCoeff[1]*y_nonzero + laneCoeff[2] + LANE_WINDOW_WIDTH)))

    # Fit the polynomial to the new indices
    xs = x_nonzero[laneIndices]
    ys = y_nonzero[laneIndices]
    if len(xs) > 0 and len(ys) > 0:
        laneCoeff = np.polyfit(ys, xs, 2)
    else:
        laneCoeff = []
    return laneCoeff, xs, ys

def calculateCurve(coeff, yEval):
    # Convert the valid index values to real world to fit
    # a polynomial to use to calculate the radius
    radius_m = 0
    if len(leftCoeff) ==3 and len(rightCoeff) == 3:
        ys = np.linspace(0, IMAGE_HEIGHT-1, IMAGE_HEIGHT)
        xs = coeff[0]*ys**2 + coeff[1]*ys + coeff[2]

        worldCoeff = np.polyfit(ys*M_PER_PIXEL_y, xs*M_PER_PIXEL_x, 2)

        numerator = (1+(2*worldCoeff[0]*yEval*M_PER_PIXEL_y + worldCoeff[1])**2)**1.5
        radius_m = numerator/np.absolute(2*worldCoeff[0])
    return radius_m

def estimateCenterOffset(leftCoeff, rightCoeff):
    y_eval = IMAGE_HEIGHT-1
    xLeft = leftCoeff[0]*y_eval**2 + leftCoeff[1]*y_eval + leftCoeff[2]
    xRight = rightCoeff[0]*y_eval**2 + rightCoeff[1]*y_eval + rightCoeff[2]
    cameraPosition = IMAGE_WIDTH/2
    midpoint = (xRight-xLeft)/2 + xLeft
    offset = (cameraPosition - midpoint)*M_PER_PIXEL_x
    return offset

class LaneLine():
    def __init__(self, isLeft, printDebug=False):
        self.isLeft = isLeft
        self.init = False
        self.detected = False
        self.xLatest = []
        self.yLatest = []
        self.radius_m = 0
        self.radiusG = .80 #Exponential smoothing used to track radius

        #calculation helpers
        self.xMin = 0
        self.xMax = 0
        self.yEval = 0

        #polynomial coefficients
        self.coeff = []
        self.coeffDelta = np.array([0,0,0], dtype='float')

        #Variable to display for testing
        self.printDebug = printDebug
        return

    def initialize(self, coeff, xs, ys):
        self.xLatest = xs
        self.yLatest = ys
        self.coeff = coeff
        self.radius_m = calculateCurve(coeff, self.yEval)

        lExpected = (IMAGE_WIDTH/2) - LANE_HALF_W_pixel
        rExpected = (IMAGE_WIDTH/2) + LANE_HALF_W_pixel
        xExpected = lExpected if self.isLeft else rExpected
        self.xMin = xExpected - LANE_WINDOW_WIDTH/2
        self.xMax = xExpected + LANE_WINDOW_WIDTH/2

        self.yEval = IMAGE_HEIGHT-1
        self.init = True
        return

    def reset(self):
        self.init = False
        self.detected = False
        self.xLatest = []
        self.yLatest = []
        self.coeff = []
        self.coeffDelta = np.array([0,0,0], dtype='float')
        self.radius_m = 0
        return

    def isValid(self, coeff):
        x = coeff[0]*self.yEval**2 + coeff[1]*self.yEval + coeff[2]
        valid = (self.xMin <= x <= self.xMax)
        return valid

    def update(self, coeff, xs, ys):
        if self.isValid(coeff):
            self.coeffDelta = np.absolute(np.subtract(self.coeff, coeff))
            self.coeff = np.divide(np.add(self.coeff, coeff),2)
            radius = calculateCurve(self.coeff, self.yEval)
            self.radius_m = self.radiusG * radius + (1-self.radiusG) * self.radius_m
        else:
            self.detected = False
            if self.printDebug:print('\tINVALID m')
        self.xLatest = xs
        self.yLatest = ys
        return

    def findInImageInitial(self, img):
        xLeft, xRight = findLaneStarts(img)
        startL, startR = findLaneStarts(img)
        xSeed = startL if self.isLeft else startR
        coeff, xs, ys = findInitialLaneInImage(img, xSeed)
        return coeff, xs, ys

    def findInImage(self, img):
        if self.printDebug:print("Lane {}".format(('Left' if self.isLeft else 'Right')))
        if self.init == False:
            coeff, xs, ys = self.findInImageInitial(img)
            self.initialize(coeff, xs, ys)
        else:
            coeff, xs, ys = findLaneInImage(img, self.coeff)
            self.detected = (len(xs) > 0)
            if self.detected:
                if self.printDebug:print("\tDetected")
                self.update(coeff, xs, ys)
        return

    def adjust(self, other):
        # Modify the values using information from the other line
        sign = -1 if self.isLeft else 1
        maxDelta = np.amax(self.coeffDelta)
        otherDelta = np.amax(other.coeffDelta)
        if other.detected and (not self.detected or (maxDelta > otherDelta)):
            if self.printDebug:print("Adjust {}".format(('Left' if self.isLeft else 'Right')))
            xNew = np.concatenate((other.xLatest+(sign*LANE_WIDTH_pixel), self.xLatest))
            yNew = np.concatenate((other.yLatest, self.yLatest))
            if len(xNew) > 0 and len(yNew) > 0:
                newCoeff = np.polyfit(yNew, xNew, 2)
            else:
                newCoeff = []
            self.update(newCoeff, xNew, yNew)
        return

def createLaneDrawing(birdsEyeImg, undistortedImg, leftLane, rightLane):
    #Create the green shape for highlighting the lane
    birdsEyeZero = np.zeros_like(birdsEyeImg).astype(np.uint8)
    birdsEyeColor = np.dstack((birdsEyeZero, birdsEyeZero, birdsEyeZero))

    xL, xR, y = plotCreateSecondOrderXY(leftLane.coeff, rightLane.coeff)
    pointsL = np.array([np.transpose(np.vstack([xL, y]))])
    pointsR = np.array([np.flipud(np.transpose(np.vstack([xR, y])))])
    pts = np.hstack((pointsL, pointsR))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(birdsEyeColor, np.int_([pts]), (0,255, 0))
    laneImg = warpFromBirdsEye(birdsEyeColor)

    # Combine the drawing with the undistorted image
    resultImg = cv2.addWeighted(undistortedImg, 1, laneImg, 0.3, 0)

    # Write the radius and offset
    textRadius = "Radius ~ {:.2f} km".format((leftLane.radius_m+rightLane.radius_m)/2000)
    resultImg = cv2.putText(resultImg, textRadius, (10,100), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)

    carOffset = estimateCenterOffset(leftLane.coeff, rightLane.coeff)
    textOffsets = "Car Offset ~ {:.2f} m from Center".format(carOffset)
    resultImg = cv2.putText(resultImg, textOffsets, (10,200), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)

    return resultImg

# ## Lane Finding Pipeline ##
# Putting together the functions definded above into a pipeline
# 1. Distortion correction
# 2. Binary image threshold creation
# 3. Perspective transformation
# 4. Lane line identification
# 5. Curvature and location estimation
# 6. Image markup
def findLanes(img, leftLane, rightLane):

    undistorted = undistortImage(img)
    binary = combineThresholds(undistorted)
    warped = warpToBirdsEye(binary)

    leftLane.findInImage(warped)
    rightLane.findInImage(warped)
    leftLane.adjust(rightLane)
    rightLane.adjust(leftLane)

    resultImg = createLaneDrawing(warped, undistorted, leftLane, rightLane)
    return resultImg
