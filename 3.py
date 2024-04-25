import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import sys
import xpc
import PID
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

update_interval = 100 # seconds
start = datetime.now()
last_update = start

# Function to calculate pitch angle
def calculate_pitch(edge_coordinates, w, h):
    m = (edge_coordinates[-1][1] - edge_coordinates[0][1]) / (edge_coordinates[-1][0] - edge_coordinates[0][0])
    c = edge_coordinates[0][1] - m * edge_coordinates[0][0]
    midpoint = w / 2
    y_offset = m * midpoint + c - h / 2
    pitch = (y_offset / (h / 2)) * 100
    return pitch

# Function to calculate pitch and roll based on detected horizon
def calculate_roll(image, horizon_line, center_vertical_line, center_horizontal_line):
    height, width, _ = image.shape

    # Extracting coordinates from lines
    x1, y1, x2, y2 = horizon_line
    cx1, cy1, cx2, cy2 = center_vertical_line
    hx1, hy1, hx2, hy2 = center_horizontal_line

    # Calculate pitch with respect to vertical center line
    #pitch = np.arctan2((y2 - y1), (x2 - x1)) * (180 / np.pi)
    #pitch_center = np.arctan2((cy2 - cy1), (cx2 - cx1)) * (180 / np.pi)
    #pitch_relative = pitch - pitch_center

    # Calculate roll with respect to horizontal center line
    roll = np.arctan2((hy2 - hy1), (hx2 - hx1)) * (180 / np.pi)
    roll_center = np.arctan2((y2 - y1), (x2 - x1)) * (180 / np.pi)
    roll_relative = roll - roll_center

    return  roll_relative

# Capture a cropped region of the screen
x, y, w, h = 100, 150, 700, 900  # Adjust these values based on your specific region of interest

# defining the initial PID values
P = 0.1 # PID library default = 0.2
I = P/10 # default = 0
D = 0 # default = 0

# initializing both PID controllers
roll_PID = PID.PID(P, I, D)
pitch_PID = PID.PID(P, I, D)
altitude_PID = PID.PID(P, I, D)  # Initialize altitude PID controller

# setting the desired values
# roll = 0 means wings level
desired_roll = 0
desired_pitch = 0
desired_altitude = 1000  # Desired altitude in feet

# setting the PID set points with our desired values
roll_PID.SetPoint = desired_roll
pitch_PID.SetPoint = desired_pitch
altitude_PID.SetPoint = desired_altitude  # Set altitude PID set point

# Lists to store roll values before and after PID
roll_before_PID_list = []
roll_after_PID_list = []

# Lists to store pitch values before and after PID
pitch_before_PID_list = []
pitch_after_PID_list = []

# Lists to store altitude values before and after PID
altitude_before_PID_list = []
altitude_after_PID_list = []

# Lists to store horizon line coordinates
horizon_x_values = []
horizon_y_values = []

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
            kernel = np.ones((10, 10), np.uint8)
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
                horizon_x_values.append(edge_coordinates[0][0])
                horizon_x_values.append(edge_coordinates[-1][0])
                horizon_y_values.append(edge_coordinates[0][1])
                horizon_y_values.append(edge_coordinates[-1][1])
                
                # Define center lines
                center_vertical_line = (w // 2, 0, w // 2, h)
                center_horizontal_line = (0, h // 2, w, h // 2)

                # Calculate pitch and roll
                roll = calculate_roll(frame, horizon_line, center_vertical_line, center_horizontal_line)

                # Get the simulated pitch value before PID
                pitch_before = client.getPOSI()[3]

                # Update PID controller with calculated roll value
                roll_PID.update(roll)

                # Get the PID output for roll
                new_ail_ctrl = roll_PID.output

                # Get the simulated pitch value after PID
                pitch_after =  client.getPOSI()[3]

                # Append roll values to lists
                roll_before_PID_list.append(roll)
                roll_after_PID_list.append(roll_PID.output)

                # Print roll, pitch, and altitude information before PID
                print(f"Roll before PID: {roll:.2f}, Pitch before PID: {pitch_before:.2f}")

                # Get the simulated pitch value
                pitch = calculate_pitch(edge_coordinates, w, h)  # Assuming pitch is the 4th value in the returned list

                # Update PID controller with simulated pitch value
                pitch_PID.update(pitch)

                # Get the PID output for pitch
                new_ele_ctrl = pitch_PID.output

                # Append pitch values to lists
                pitch_before_PID_list.append(pitch)
                pitch_after_PID_list.append(pitch_PID.output)
                
                # Get the simulated altitude value
                altitude_before = client.getPOSI()[2]  # Assuming altitude is the 3rd value in the returned list

                # Update PID controller with simulated altitude value
                altitude_PID.update(altitude_before)

                # Get the PID output for altitude
                new_alt_ctrl = altitude_PID.output
                
                # Append altitude values to lists
                altitude_before_PID_list.append(altitude_before)
                altitude_after_PID_list.append(altitude_PID.output)

                # Print roll, pitch, and altitude information after PID
                print(f"Roll after PID: {roll_PID.output:.2f}, Pitch after PID: {pitch_after:.2f}, Altitude: {altitude_before:.2f}")

                # Send control input to X-Plane
                ctrl = [new_ele_ctrl, new_ail_ctrl, 0.0, new_alt_ctrl]  # Adjust the control array to include altitude control
                client.sendCTRL(ctrl)

               

                # Draw horizon line
                cv2.line(frame, (horizon_line[0], horizon_line[1]), (horizon_line[2], horizon_line[3]), (0, 0, 255), 2)

                # Calculate and display pitch angle
                pitch = calculate_pitch(edge_coordinates, w, h)
                cv2.putText(frame, f'Pitch: {pitch:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # Display roll value
                cv2.putText(frame, f'Roll: {roll:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # Show the frame
                cv2.imshow('Frame', frame)
                cv2.imshow('Edge', edges)
                cv2.waitKey(1)

                # Break the loop on 'z' key press
                if cv2.waitKey(1) & 0xFF == ord('z'):
                    break

        # Plotting the roll before and after PID
        #plt.plot(roll_before_PID_list, label='Roll before PID')
        #plt.plot(roll_after_PID_list, label='Roll after PID')
        #plt.xlabel('Frame')
        #plt.ylabel('Roll Angle')
        #plt.title('Roll Angle Comparison')
        #plt.legend()
        #plt.show()

        # Plotting the pitch before and after PID
        #plt.plot(pitch_before_PID_list, label='Pitch before PID')
        #plt.plot(pitch_after_PID_list, label='Pitch after PID')
        #plt.xlabel('Frame')
        #plt.ylabel('Pitch Angle')
        #plt.title('Pitch Angle Comparison')
        #plt.legend()
        #plt.show()

        # Plotting the altitude before and after PID
        #plt.plot(altitude_before_PID_list, label='Altitude before PID')
        #plt.plot(altitude_after_PID_list, label='Altitude after PID')
        #plt.xlabel('Frame')
        #plt.ylabel('Altitude (feet)')
        #plt.title('Altitude Comparison')
        #plt.legend()
        #plt.show()

        # Plotting the horizon line
        #plt.plot(horizon_x_values, horizon_y_values)
        #plt.xlabel('X-coordinate')
        #plt.ylabel('Y-coordinate')
        #plt.title('Horizon Line')
        #plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        #plt.show()

if __name__ == "__main__":
    monitor()
