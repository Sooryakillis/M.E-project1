import xpc
import PID
from datetime import datetime, timedelta
import pyqtgraph as pg
import sys
from PyQt5.QtGui import QGuiApplication
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

def normalize(value, min=-1, max=1):
    if value > max:
        return max
    elif value < min:
        return min
    else:
        return value

update_interval = 0.050  # seconds, 0.05 = 20 Hz
start = datetime.now()
last_update = start

# defining the initial PID values
P = 0.1  # PID library default = 0.2
I = P / 10  # default = 0
D = 0  # default = 0

# initializing PID controllers
roll_PID = PID.PID(P, I, D)
pitch_PID = PID.PID(P, I, D)
altitude_PID = PID.PID(P, I, D)

# setting the desired values
desired_roll = 0
desired_pitch = 0
desired_altitude = 5000

# setting the PID set points with our desired values
roll_PID.SetPoint = desired_roll
pitch_PID.SetPoint = desired_pitch
altitude_PID.SetPoint = desired_altitude

x_axis_counters = []
roll_history = []
pitch_history = []
altitude_history = []
roll_setpoint_history = []
pitch_setpoint_history = []
altitude_setpoint_history = []
plot_array_max_length = 1000
i = 1

app = pg.mkQApp("python xplane autopilot monitor")
win = pg.GraphicsLayoutWidget(show=True)
win.resize(1000, 600)
win.setWindowTitle("XPlane autopilot system control")

p1 = win.addPlot(title="roll", row=0, col=0)
p2 = win.addPlot(title="pitch", row=1, col=0)
p3 = win.addPlot(title="altitude", row=2, col=0)

p1.showGrid(y=True)
p2.showGrid(y=True)
p3.showGrid(y=True)

x, y, w, h = 200, 200, 1400, 900

DREFs = ["sim/cockpit2/gauges/indicators/airspeed_kts_pilot",
         "sim/cockpit2/gauges/indicators/heading_electric_deg_mag_pilot",
         "sim/flightmodel/failures/onground_any",
         "sim/flightmodel/misc/h_ind"]

def monitor():
    global i
    global last_update
    with xpc.XPlaneConnect() as client:
        while True:
            if (datetime.now() > last_update + timedelta(milliseconds=update_interval * 1000)):
                last_update = datetime.now()
                screenshot = pyautogui.screenshot(region=(x, y, w, h))
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 10, 20)
                kernel = np.ones((5, 5), np.uint8)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                edge_coordinates = []
                for j in range(w):
                    col_edges = np.where(edges[:, j] > 0)[0]
                    if len(col_edges) > 0:
                        edge_coordinates.append((j, col_edges[0]))

                if len(edge_coordinates) >= 2:
                    horizon_line = (edge_coordinates[0][0], edge_coordinates[0][1], edge_coordinates[-1][0],
                                    edge_coordinates[-1][1])
                    center_vertical_line = (w // 2, 0, w // 2, 1000)
                    center_horizontal_line = (0, h // 2, w, h // 2)

                                    # Draw horizon line
                cv2.line(frame, (horizon_line[0], horizon_line[1]), (horizon_line[2], horizon_line[3]), (0, 0, 255), 2)

                # Calculate and display pitch angle
                pitch = calculate_pitch(edge_coordinates, w, h)
                cv2.putText(frame, f'Pitch: {pitch:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                roll = calculate_pitch_and_roll(frame, horizon_line, center_vertical_line,
                                                     center_horizontal_line)
                # Display roll value
                cv2.putText(frame, f'Roll: {roll:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)



                # Show the frame
                cv2.imshow('Frame', frame)
                #cv2.imshow('edge', edges)
              

                posi = client.getPOSI()
                ctrl = client.getCTRL()
                multi_DREFs = client.getDREFs(DREFs)

                current_roll = calculate_pitch_and_roll(frame, horizon_line, center_vertical_line, center_horizontal_line) 
                current_pitch = posi[3]
                current_hdg = multi_DREFs[1][0]
                current_altitude = multi_DREFs[3][0]
                current_asi = multi_DREFs[0][0]
                onground = multi_DREFs[2][0]

                pg.QtGui.QGuiApplication.processEvents()

                altitude_PID.update(current_altitude)
                pitch_PID.SetPoint = normalize(altitude_PID.output, min=-15, max=10)
                roll_PID.update(current_roll)
                pitch_PID.update(current_pitch)

                new_ail_ctrl = normalize(roll_PID.output)
                new_ele_ctrl = normalize(pitch_PID.output)

                if len(x_axis_counters) > plot_array_max_length:
                    x_axis_counters.pop(0)
                    roll_history.pop(0)
                    roll_setpoint_history.pop(0)
                    pitch_history.pop(0)
                    pitch_setpoint_history.pop(0)
                    altitude_history.pop(0)
                    altitude_setpoint_history.pop(0)

                    x_axis_counters.append(i)
                    roll_history.append(current_roll)
                    roll_setpoint_history.append(desired_roll)
                    pitch_history.append(current_pitch)
                    pitch_setpoint_history.append(pitch_PID.SetPoint)
                    altitude_history.append(0)
                    altitude_setpoint_history.append(desired_altitude)
                else:
                    x_axis_counters.append(i)
                    roll_history.append(current_roll)
                    roll_setpoint_history.append(desired_roll)
                    pitch_history.append(current_pitch)
                    pitch_setpoint_history.append(pitch_PID.SetPoint)
                    altitude_history.append(0)
                    altitude_setpoint_history.append(desired_altitude)
                i = i + 1

                p1.plot(x_axis_counters, roll_history, pen=0, clear=True)
                p1.plot(x_axis_counters, roll_setpoint_history, pen=1)

                p2.plot(x_axis_counters, pitch_history, pen=0, clear=True)
                p2.plot(x_axis_counters, pitch_setpoint_history, pen=1)

                p3.plot(x_axis_counters, altitude_history, pen=0, clear=True)
                p3.plot(x_axis_counters, altitude_setpoint_history, pen=1)

                ctrl = [new_ele_ctrl, new_ail_ctrl, 0.0, -998]
                client.sendCTRL(ctrl)

                

                output = f"current values --    roll: {current_roll: 0.3f},  pitch: {current_pitch: 0.3f}"
                output = output + "\n" + f"PID outputs    --    roll: {roll_PID.output: 0.3f},  pitch: {pitch_PID.output: 0.3f}"
                output = output + "\n" 
                print(output)

                

if __name__ == "__main__":
    monitor()
a = 123