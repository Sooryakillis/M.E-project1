import cv2
import numpy as np

def find_line_center(line):
    x1, y1, x2, y2 = line
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def calculate_angle(line1, line2):
    center1 = find_line_center(line1)
    center2 = find_line_center(line2)
    angle = np.arctan2(center2[1] - center1[1], center2[0] - center1[0]) * 180 / np.pi
    return angle

def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return lines

def draw_lines(image, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

image1 = cv2.imread('3.jpg')
image2 = cv2.imread('4.jpg')

lines1 = detect_lines(image1)
lines2 = detect_lines(image2)

angle = calculate_angle(lines1[0][0], lines2[0][0])

print("Angle between the lines:", angle)

# Visualize the lines on the images
draw_lines(image1, lines1)
draw_lines(image2, lines2)

cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
