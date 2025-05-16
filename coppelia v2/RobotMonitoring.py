# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:35:32 2025

@author: amars
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def visualize_monitoring(monitoring, target, title_info, output_folder="plots"):
    
    pre_filename = output_folder
    
    # Extract monitoring data
    x_vals = [monitor[0] for monitor in monitoring][1:]
    y_vals = [monitor[1] for monitor in monitoring][1:]
    theta_vals = [monitor[2] for monitor in monitoring][1:]
    error_x = [monitor[3] for monitor in monitoring][1:]
    error_y = [monitor[4] for monitor in monitoring][1:]
    error_theta = [monitor[5] for monitor in monitoring][1:]
    q1_vals = [monitor[6] for monitor in monitoring][1:]
    q2_vals = [monitor[7] for monitor in monitoring][1:]
    d_vals = [monitor[8] for monitor in monitoring][1:]

    ###### Plot Robot's Path ######
    # Initialize segments
    segments_x = [[]]
    segments_y = [[]]
    manual_change = []

    # Iterate through the points and split into segments
    for i in range(1, len(x_vals)):
        segments_x[-1].append(x_vals[i-1])
        segments_y[-1].append(y_vals[i-1])
        
        threshold = 0.4
        if abs(x_vals[i] - x_vals[i-1]) > threshold or abs(y_vals[i] - y_vals[i-1]) > threshold:
            segments_x.append([])  # Start a new segment
            segments_y.append([])
            manual_change.append(i)

    # Add the last point to the final segment
    segments_x[-1].append(x_vals[-1])
    segments_y[-1].append(y_vals[-1])

    # Create output folder
    os.makedirs("Curves/" + output_folder, exist_ok=True)

    # Plot and save the robot's path
    plt.figure(dpi=300)
    for seg_x, seg_y in zip(segments_x, segments_y):
        plt.plot(seg_x, seg_y, color='blue')  # Plot each segment
        plt.scatter(seg_x[0], seg_y[0], color='green')  # Start point of each segment
    
    plt.scatter(target[0], target[1], color="red", label="Objective")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    rect = Rectangle((-2.5, -2.5), 5, 5, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect)
    plt.title("Robot's trajectory " + title_info)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    fig_name = pre_filename + "_robot_path.png"
    fig_path = f"Curves/{output_folder}/" + fig_name
    plt.savefig(fig_path)
    plt.close()
    
    ###### Plot Robot trajectory for each coordinate ######
    plt.figure(dpi=300)
    plt.plot(x_vals)
    for i in manual_change:
        plt.axvline(x=i, color='r', linestyle='dotted')
    plt.title("Coordinates in X " + title_info)
    plt.xlabel("Samples")
    plt.ylabel("Position (m)")
    fig_name = pre_filename + "_x_coordinates.png"
    fig_path = f"Curves/{output_folder}/" + fig_name
    plt.savefig(fig_path)
    plt.close()
    
    plt.figure(dpi=300)
    plt.plot(y_vals)
    for i in manual_change:
        plt.axvline(x=i, color='r', linestyle='dotted')
    plt.title("Coordinates in Y " + title_info)
    plt.xlabel("Samples")
    plt.ylabel("Position (m)")
    fig_name = pre_filename + "_y_coordinates.png"
    fig_path = f"Curves/{output_folder}/" + fig_name
    plt.savefig(fig_path)
    plt.close()
    
    plt.figure(dpi=300)
    plt.plot(theta_vals)
    for i in manual_change:
        plt.axvline(x=i, color='r', linestyle='dotted')
    plt.title("Coordinates in Theta " + title_info)
    plt.xlabel("Samples")
    plt.ylabel("Position (rad)")
    fig_name = pre_filename + "_theta_coordinates.png"
    fig_path = f"Curves/{output_folder}/" + fig_name
    plt.savefig(fig_path)
    plt.close()
    
    ###### Plot Motor Speeds ######
    plt.figure(dpi=300)
    plt.plot(q1_vals, label='q1')
    plt.plot(q2_vals, label='q2')
    for i in manual_change:
        plt.axvline(x=i, color='r', linestyle='dotted')
    plt.title("Motor speeds " + title_info)
    plt.xlabel("Samples")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    fig_name = pre_filename + "_motor_speeds.png"
    fig_path = f"Curves/{output_folder}/" + fig_name
    plt.savefig(fig_path)
    plt.close()
    
    ###### Plot Drift Speed ######
    plt.figure(dpi=300)
    plt.plot(d_vals)
    for i in manual_change:
        plt.axvline(x=i, color='r', linestyle='dotted')
    plt.title("Drift speed " + title_info)
    plt.xlabel("Samples")
    plt.ylabel("Speed (m)/s)")
    fig_name = pre_filename + "_drift_speed.png"
    fig_path = f"Curves/{output_folder}/" + fig_name
    plt.savefig(fig_path)
    plt.close()

    ###### Plot Loss ######
    losses = {
        "Loss in x": np.array(error_x)**2/(2.5**2),
        "Loss in y": np.array(error_y)**2/(2.5**2),
        "Loss in theta": np.array(error_theta)**2/(np.pi**2),
    }

    for loss_title, loss_values in losses.items():
        plt.figure(dpi=300)
        plt.plot(loss_values)
        for i in manual_change:
            plt.axvline(x=i, color='r', linestyle='dotted')
        plt.ylim(bottom=0, top=1)
        plt.title(loss_title + " " + title_info)
        plt.xlabel("Samples")
        plt.ylabel("Loss")
        file_name = pre_filename + "_" + loss_title.replace(" ", "_").lower() + ".png"
        plt.savefig(f"Curves/{output_folder}/{file_name}")
        plt.close()

    ###### Combined Plot with Separate Subplots ######
    fig, axs = plt.subplots(6, 1, figsize=(8, 15), dpi=300)

    # 1. Robot's Path
    for seg_x, seg_y in zip(segments_x, segments_y):
        axs[0].plot(seg_x, seg_y, color='blue')  # Plot each segment
        axs[0].scatter(seg_x[0], seg_y[0], color='green')  # Start point of each segment
    axs[0].scatter(target[0], target[1], color="red", label="Objective")
    axs[0].set_xlim(-3, 3)
    axs[0].set_ylim(-3, 3)
    axs[0].set_title("Robot's Trajectory " + title_info)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    # 2. Motor Speeds
    axs[1].plot(q1_vals, label='q1 (left)')
    axs[1].plot(q2_vals, label='q2 (right)')
    for i in manual_change:
        axs[1].axvline(x=i, color='r', linestyle='dotted')
    axs[1].set_title("Motor Speeds " + title_info)
    axs[1].set_xlabel("Samples")
    axs[1].set_ylabel("Speed (m/s)")
    axs[1].legend()
    
    # 3. Drift
    axs[2].plot(d_vals)
    for i in manual_change:
        axs[2].axvline(x=i, color='r', linestyle='dotted')
    axs[2].set_title("Drifting speed " + title_info)
    axs[2].set_xlabel("Samples")
    axs[2].set_ylabel("Speed (m/s)")

    # 4. Loss in x
    axs[3].plot(np.array(x_vals)**2/(2.5*3))
    for i in manual_change:
        axs[3].axvline(x=i, color='r', linestyle='dotted')
    axs[3].set_ylim(bottom=0, top=1)
    axs[3].set_title("Loss in X " + title_info)
    axs[3].set_xlabel("Samples")
    axs[3].set_ylabel("Loss")

    # 5. Loss in y
    axs[4].plot(np.array(y_vals)**2/(2.5**2))
    for i in manual_change:
        axs[4].axvline(x=i, color='r', linestyle='dotted')
    axs[4].set_ylim(bottom=0, top=1)
    axs[4].set_title("Loss in Y " + title_info)
    axs[4].set_xlabel("Samples")
    axs[4].set_ylabel("Loss")

    # 6. Loss in theta
    axs[5].plot(np.array(theta_vals)**2/(np.pi**2))
    for i in manual_change:
        axs[5].axvline(x=i, color='r', linestyle='dotted')
    axs[5].set_ylim(bottom=0, top=1)
    axs[5].set_title("Loss in Theta " + title_info)
    axs[5].set_xlabel("Samples")
    axs[5].set_ylabel("Loss")
    

    plt.tight_layout()
    fig_name = pre_filename + "_preview.png"
    fig_path = "Curves/" + fig_name
    plt.savefig(fig_path)
    plt.close()
    
    np.save("Curves/" + output_folder + "/" + pre_filename+"_monitoring", monitoring)

    print(f"Plots saved in the '{output_folder}' folder.")
