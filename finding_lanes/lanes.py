import cv2
import numpy as np
import matplotlib.pyplot as plt # version 3.1.1

# Canny function calculates derivate in both x and y direction, therefore changes in intensity could be calculated. 
# Larger derivatives -> High intensity(sharp changes) , Smaller derivatives -> Low intensity(shallow changes) 
def canny(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Convert the image color to grayscale
	blur = cv2.GaussianBlur(gray_image, (5,5), 0) # Reduce noise from the image
	canny = cv2.Canny(blur,50,150)
	return canny

# Where we need our line to be placed at image
def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1 * (3/5))
	x1 = int((y1 - intercept)/ slope)
	x2 =  int((y2 - intercept)/ slope)
	return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
	left_fit = []
	right_fit = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters = np.polyfit((x1, x2),(y1, y2),1) # It will fit the polynomial and return the intercept and slope
		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
			left_fit.append((slope, intercept))
		else:
			right_fit.append((slope, intercept))
	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	return np.array([left_line, right_line])

def display_lines(image, lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for x1, y1, x2, y2 in lines:
			cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
	return line_image

def region_of_interest(image):
	height = image.shape[0]
	polygons = np.array([
		[(200, height),(1100, height),(550, 250)]
		])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, polygons, 255) # Fill poly function deals with multiple polygon
	masked_image = cv2.bitwise_and(image, mask) # Bitwise operation between canny image and mask image
	return masked_image

# Reading images using openCV
# imageFile = cv2.imread("C:\\Users\\Dapplogix\\Documents\\self_driving\\datasets\\Image\\test_image.jpg")

# # Converting the image intp gray scale
# lane_image = np.copy(imageFile) # Immutable copy

# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines =  cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # HoughLinesP detects the straight line going through an image according to number of votes
# averaged_lines = average_slope_intercept(lane_image, lines) 
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) 
# cv2.imshow("results", combo_image)
# cv2.waitKey(0) # wait 0 will show the image infinitely, waits for 0 ms

cap = cv2.VideoCapture("C:\\Users\\Dapplogix\\Documents\\self_driving\\datasets\\test2.mp4\\test2.mp4")

# Get the Default resolutions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and filename.

while(cap.isOpened()):
	_, frame = cap.read()
	canny_image = canny(frame)
	cropped_image = region_of_interest(canny_image)
	lines =  cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # HoughLinesP detects the straight line going through an image according to number of votes
	averaged_lines = average_slope_intercept(frame, lines) 
	line_image = display_lines(frame, averaged_lines)
	combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) 
	cv2.imshow("results", combo_image)
	# When the below two will be true and will press the 'q' on our keyboad, we will break out from the loop
	if cv2.waitKey(1) & 0xFF == ord('q'): # wait 0 will wait for infinitely between each frames. 1ms will wait for the specified time only between each frames 	
		break

cap.release() # close the video file
cv2.destroyAllWindows() # destroy all the windows that is currently on