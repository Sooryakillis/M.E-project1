import cv2
import numpy as np
import pygetwindow as gw
import pyautogui

# Function to calculate pitch and roll based on detected horizon
def calculate_pitch_and_roll(image, horizon_line):
    height, width, _ = image.shape

    x1, y1, x2, y2 = horizon_line

    pitch = np.arctan2((y2 - y1), width) * (180 / np.pi)
    roll = np.arctan2((x2 - x1), height) * (180 / np.pi)

    return pitch, roll

while True:
    # Capture a cropped region of the screen
    x, y, w, h = 100, 150, 1500, 1000  # Adjust these values based on your specific region of interest
    screenshot = pyautogui.screenshot(region=(x, y, w, h))

    # Convert the screenshot to a NumPy array
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #cv2.imshow('blur', frame)


    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(gray,9,50,50)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 10, 20)

    # Apply dilation followed by erosion (closing operation) to clear unwanted edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Initialize variables to store coordinates of the edges
    edge_coordinates = []

    # Scan edges row-wise
    for i in range(h):
     row_edges = np.where(edges[i, :] > 0)[0]
    if len(row_edges) > 0:
     edge_coordinates.append((row_edges[0], i))

    # Scan edges column-wise
    for j in range(w):
        col_edges = np.where(edges[:, j] > 0)[0]
        if len(col_edges) > 0:
            edge_coordinates.append((j, col_edges[0]))

    # Draw lines based on the coordinates
    for coord in edge_coordinates:
        cv2.circle(frame, coord, 2, (0, 255, 0), -1)

    # Display both edge detection and line detection
    cv2.imshow('Edge and Line Detection', frame)
    cv2.imshow('Edge Detection', edges)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

# Release resources
cv2.destroyAllWindows()
