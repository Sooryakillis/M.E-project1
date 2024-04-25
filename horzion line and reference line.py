import cv2
import numpy as np
import pygetwindow as gw
import pyautogui

# Function to calculate pitch and roll based on detected horizon
def calculate_pitch_and_roll(image, horizon_line, center_vertical_line, center_horizontal_line):
    height, width, _ = image.shape

    # Extracting coordinates from lines
    x1, y1, x2, y2 = horizon_line
    cx1, cy1, cx2, cy2 = center_vertical_line
    hx1, hy1, hx2, hy2 = center_horizontal_line

    # Calculate pitch with respect to vertical center line
    pitch = np.arctan2((y2 - y1), (x2 - x1)) * (180 / np.pi)
    pitch_center = np.arctan2((cy2 - cy1), (cx2 - cx1)) * (180 / np.pi)
    pitch_relative = pitch - pitch_center

    # Calculate roll with respect to horizontal center line
    roll = np.arctan2((hy2 - hy1), (hx2 - hx1)) * (180 / np.pi)
    roll_center = np.arctan2((y2 - y1), (x2 - x1)) * (180 / np.pi)
    roll_relative = roll - roll_center

    return pitch_relative, roll_relative

while True:
    # Capture a cropped region of the screen
    x, y, w, h = 100, 150, 1400, 1000  # Adjust these values based on your specific region of interest
    screenshot = pyautogui.screenshot(region=(x, y, w, h))

    # Convert the screenshot to a NumPy array
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 10, 20)

    # Apply dilation followed by erosion (closing operation) to clear unwanted edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Initialize variables to store coordinates of the edges
    edge_coordinates = []

    # Scan edges column-wise
    for j in range(w):
        col_edges = np.where(edges[:, j] > 0)[0]
        if len(col_edges) > 0:
            edge_coordinates.append((j, col_edges[0]))


    # Draw lines based on the coordinates
    for coord in edge_coordinates:
        cv2.circle(frame, coord, 1, (1, 1, 1), -1)
        #cv2.imshow('Circle', frame)

    # Calculate horizon line (for demonstration purposes, here we just use the first and last edge points)
    if len(edge_coordinates) >= 2:
        horizon_line = (edge_coordinates[0][0], edge_coordinates[0][1], edge_coordinates[-1][0], edge_coordinates[-1][1])
        
        # Define center lines
        center_vertical_line = (w // 2, 0, w // 2, h)
        center_horizontal_line = (0, h // 2, w, h // 2)

        # Calculate pitch and roll
        pitch, roll = calculate_pitch_and_roll(frame, horizon_line, center_vertical_line, center_horizontal_line)
        print("Pitch:", pitch, " Roll:", roll)

        # Draw horizon line
        cv2.line(frame, (horizon_line[0], horizon_line[1]), (horizon_line[2], horizon_line[3]), (0, 0, 255), 2)

    # Display both edge detection and line detection
    cv2.imshow('Horizon Line Detection', frame)
    cv2.imshow('Edge Detection', edges)

    # Add vertical and horizontal lines
    # Vertical line at the center
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 3)
    # Horizontal line at the center
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 3)
    # Display the frame with lines
    cv2.imshow('Vertical and Horizontal Lines', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

# Release resources
cv2.destroyAllWindows()
