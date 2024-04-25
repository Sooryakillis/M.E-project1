import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import sys
import xpc
import PID
from datetime import datetime, timedelta
from openpyxl import Workbook

# Initialize workbook and sheet
wb = Workbook()
sheet = wb.active
sheet.append(["Time", "Roll", "Pitch", "XplaneRoll", "XplanePitch"])

# Function to write data to Excel sheet
def write_to_excel(time, roll, pitch, XplaneRoll, XplanePitch ):
    sheet.append([time, roll, pitch, XplaneRoll, XplanePitch])
    wb.save("roll_pitch_data.xlsx")

# Function to calculate pitch angle
def calculate_pitch(edge_coordinates, w, h):
    m = (edge_coordinates[-1][1] - edge_coordinates[0][1]) / (edge_coordinates[-1][0] - edge_coordinates[0][0])
    c = edge_coordinates[0][1] - m * edge_coordinates[0][0]
    midpoint = w / 2
    y_offset = m * midpoint + c - h / 2
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

# Capture a cropped region of the screen
x, y, w, h = 100, 150, 1500, 1000  # Adjust these values based on your specific region of interest

# defining the initial PID values
P = 0.1 # PID library default = 0.2
I = P/10 # default = 0
D = 0 # default = 0

# initializing both PID controllers
roll_PID = PID.PID(P, I, D)
pitch_PID = PID.PID(P, I, D)
altitude_PID = PID.PID(P, I, D)

# setting the desired values
# roll = 0 means wings level
desired_roll = -10
desired_pitch = 10
#desired_altitude_offset = 1000  # Desired altitude offset from the horizon line

# setting the PID set points with our desired values
roll_PID.SetPoint = desired_roll
pitch_PID.SetPoint = desired_pitch
#altitude_PID.SetPoint = desired_altitude_offset

def monitor():
    with xpc.XPlaneConnect() as client:
        while True:
            # Capture the screenshot
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

            # Calculate horizon line (for demonstration purposes, here we just use the first and last edge points)
            if len(edge_coordinates) >= 2:
                horizon_line = (edge_coordinates[0][0], edge_coordinates[0][1], edge_coordinates[-1][0], edge_coordinates[-1][1])
                
                # Define center lines
                center_vertical_line = (w // 2, 0, w // 2, h)
                center_horizontal_line = (0, h // 2, w, h // 2)


                # Calculate pitch and roll
                roll = calculate_pitch_and_roll(frame, horizon_line, center_vertical_line, center_horizontal_line)

                # Update PID controller with calculated roll value
                roll_PID.update(roll)

                # Get the PID output for roll
                new_ail_ctrl = roll_PID.output

                # Calculate pitch angle
                pitch = client.getPOSI()[3]

                # Calculate desired pitch based on the pitch angle relative to the horizon line
                desired_pitch = 0   # Reverse the sign to maintain pitch 0% at horizon

                # Update PID controller with desired pitch
                pitch_PID.update(pitch)

                # Get the PID output for pitch
                new_ele_ctrl = pitch_PID.output

                XplaneRoll = client.getPOSI()[4]
                XplanePitch = client.getPOSI()[3]
                print(f"XplaneRoll: {XplaneRoll:.2f}, XplanePitch: {XplanePitch:.2f}")

                # Calculate altitude offset relative to the horizon line
                #altitude_offset = calculate_pitch(edge_coordinates, w, h)

                #Update PID controller with altitude offset
                #altitude_PID.update(altitude_offset)

                # Get the PID output for altitude
                #new_throttle_ctrl = altitude_PID.output

                # Print roll, pitch, and altitude information
                print(f"Roll: {roll:.2f}, Pitch: {pitch:.2f}")

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                write_to_excel(current_time, roll, pitch, XplaneRoll, XplanePitch)

                # Send control input to X-Plane
                ctrl = [new_ele_ctrl, new_ail_ctrl, 0.0,]
                client.sendCTRL(ctrl)

                # Draw horizon line
                cv2.line(frame, (horizon_line[0], horizon_line[1]), (horizon_line[2], horizon_line[3]), (0, 0, 255), 2)

                # Calculate and display pitch angle
                cv2.putText(frame, f'Pitch: {pitch:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # Display roll value
                cv2.putText(frame, f'Roll: {roll:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # Show the frame
                cv2.imshow('Frame', frame)
                #cv2.imshow('Edge', edges)
                cv2.waitKey(1)

                # Break the loop on 'z' key press
                if cv2.waitKey(1) & 0xFF == ord('z'):
                    break

if __name__ == "__main__":
    monitor()
