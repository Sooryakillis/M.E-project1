import cv2
import numpy as np
import pyautogui

# Function to calculate pitch angle
def calculate_pitch(edge_coordinates, w, h):
    # Calculate the center point of the image
    image_center_x = w / 2
    image_center_y = h / 2

    # Calculate the center point of the horizon line
    horizon_center_x = (edge_coordinates[-1][0] + edge_coordinates[0][0]) / 2
    horizon_center_y = (edge_coordinates[-1][1] + edge_coordinates[0][1]) / 2

    # Calculate the vertical offset of the horizon center from the image center
    y_offset = horizon_center_y - image_center_y

    # Calculate the pitch angle based on the vertical offset
    pitch = (y_offset / (h / 2)) * 100
    return pitch

# Function to calculate pitch and roll based on detected horizon
def calculate_pitch_and_roll(image, horizon_line, center_vertical_line, center_horizontal_line):
    height, width, _ = image.shape

    # Extracting coordinates from lines
    x1, y1, x2, y2 = horizon_line
    cx1, cy1, cx2, cy2 = center_vertical_line
    hx1, hy1, hx2, hy2 = center_horizontal_line

    # Calculate roll with respect to horizontal center line
    roll = np.arctan2((hy2 - hy1), (hx2 - hx1)) * (180 / np.pi)
    roll_center = np.arctan2((y2 - y1), (x2 - x1)) * (180 / np.pi)
    roll_relative = roll - roll_center

    return roll_relative

while True:
    # Capture a cropped region of the screen
    x, y, w, h = 100, 150, 1400, 1000  # Adjust these values based on your specific region of interest
    screenshot = pyautogui.screenshot(region=(x, y, w, h))

    # Convert the screenshot to a NumPy array
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Frame', gray)

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

    # Calculate horizon line (for demonstration purposes, here we just use the first and last edge points)
    if len(edge_coordinates) >= 2:
        horizon_line = (edge_coordinates[0][0], edge_coordinates[0][1], edge_coordinates[-1][0], edge_coordinates[-1][1])
        
        # Define center lines
        center_vertical_line = (w // 2, 0, w // 2, h)
        center_horizontal_line = (0, h // 2, w, h // 2)

        # Calculate pitch and roll
        roll = calculate_pitch_and_roll(frame, horizon_line, center_vertical_line, center_horizontal_line)
        cv2.putText(frame, f'Roll: {roll:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Draw horizon line
        cv2.line(frame, (horizon_line[0], horizon_line[1]), (horizon_line[2], horizon_line[3]), (0, 0, 255), 2)

        # Calculate and display pitch angle
        pitch = calculate_pitch(edge_coordinates, w, h)
        cv2.putText(frame, f'Pitch: {pitch:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display both edge detection and line detection
    cv2.imshow('Horizon Line Detection', frame)
    #cv2.imshow('Edge Detection', edges)

    # Add vertical and horizontal lines
    # Vertical line at the center
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 3)
    # Horizontal line at the center
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 3)
    # Display the frame with lines
    #cv2.imshow('Vertical and Horizontal Lines', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

# Release resources
cv2.destroyAllWindows()
