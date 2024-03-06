import os
from collections import deque
import tkinter as tk
import time
import math
import random
import csv
import pandas as pd
import numpy as np
from models import BayesRegression
from scaling_policy import ScalingPolicy, BalancedScalingPolicy
from process import transform_2d_coordinate, compute_osd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

class InstrumentTracker:
        def __init__(self, root, data_folder):
                self.root = root
                self.root.title("Instrument Tracker")

                # set logging files
                self.data_folder = data_folder
                self.param_file = f"{data_folder}/tested_params.csv"

                # Setting up game parameters
                self.latency_domain = [0.25] #[round(0.25 * i, 2) for i in range(4)]
                self.scale_domain = np.round(np.arange(0.1, 1.1, 0.1), 1)
                self.target_distance = 222
                self.target_width = 40
                self.visited_params = dict(
                        latency = [],
                        scale = []
                )

                # Create a list of all possible combinations of (x, y)
                # self.game_params = [(x, y) for x in self.latency_domain for y in self.scale_domain]
                self.prediction_input = np.array([[l, s] for l in self.latency_domain for s in self.scale_domain]).T
                self.prediction_df = pd.DataFrame({
                        'latency': self.prediction_input[0,:],
                        'scale': self.prediction_input[1,:],
                })
                self.obs_data = []

                # Initialize Models
                self.model = BayesRegression()
                self.scaling_policy = BalancedScalingPolicy(scale_domain=self.scale_domain)
                # Set prior?
                # Choose init conditions
                self.control_scale = 0.6 # TO DO: later can be replaced with optimal from prior


                # Removing unnecessary param combos
                # self.game_params.remove((0.0, 0.1))
                # # # self.game_params.remove((0.25, 0.1))
                # # # self.game_params.remove((0.5, 1.0))
                # self.game_params.remove((0.75, 1.0))

                # # If param file exists, remove already tested params, else create file and add header line
                # param_file = f"{data_folder}/tested_params.csv"
                # data_to_remove = []

                # if os.path.exists(param_file):
                #         with open(param_file, 'r') as csv_file:
                #                 csv_reader = csv.reader(csv_file)
                #                 for i, row in enumerate(csv_reader):
                #                         if i > 0:  # Skip header row
                #                                 x_value = float(row[0])
                #                                 y_value = float(row[1])
                #                                 data_to_remove.append((x_value, y_value))

                #                                 # Remove the extracted combinations from the list of tuples
                #                                 for item in data_to_remove:
                #                                         if item in self.game_params:
                #                                                 self.game_params.remove(item)
                # else:
                #         print("creating param file")
                #         with open(self.param_file, mode="w", newline="") as file:
                #                 writer = csv.writer(file)
                #                 header = ['latency', 'scaling_factor', 'target distance', 'target width']
                #                 writer.writerow(header)

                # # If all params have been tested already
                # if not self.game_params:
                #         print("All parameters have been tested already!")
                #         self.root.quit()
                #         self.root.destroy()
                #         return
                        
                        
                # # Randomize order of param combo
                # random.shuffle(self.game_params)

                # self.total_trials = len(self.game_params)
                self.trial_num = 0
                # print("Total trials left: ", total_trials)
                
                # Setting up main GUI
                self.screen_width = root.winfo_screenwidth()
                self.screen_height = root.winfo_screenheight()
                self.screen_center_x = self.screen_width // 2
                self.screen_center_y = self.screen_height // 2
                self.root.geometry(f"{self.screen_width}x{self.screen_height}")
                self.canvas = tk.Canvas(root, width=self.screen_width, height=self.screen_height, bg="white")
                self.canvas.pack()

                # button variables and flags
                self.save_button = None  # Button to save data
                self.dont_save_button = None  # Button to return to the initial screen without saving
                self.start_button = None
                self.quit_button = None
                self.save_data = False  # Flag to determine whether to save data
                self.trial_running = False
                self.warning_rectangles = {}
                self.practice_mode = False

                # Target info
                self.target_positions = []
                self.num_targets = 10
                self.target_hit = [False] * self.num_targets  # Track target hits
                self.target_shapes = []  # Store target shapes
                self.current_target = 0  # Track the current target
                self.instrument = None

                # data variables
                self.data_header = ["time", "ins_x", "ins_y", "click", "clutch"]
                self.mouse_queue = deque()
                self.game_data = []
                self.movement_data = []
                self.game_start_time = None  # Timestamp when the game started
                self.game_end_time = None  # Timestamp when the game ended
                self.trial_num = 0

                # Display Start Screen
                self.display_start_screen()
                
                # Switch to fullscreen
                self.root.bind("<f>", self.toggle_fullscreen)
                self.root.bind("<Escape>", self.end_fullscreen)
                self.fullscreen_state = True
                self.root.attributes("-fullscreen", self.fullscreen_state)
                
        
        def start_game(self):

                # Set flags
                self.trial_running = True
                self.num_clicks = 0

                self.canvas.delete("all") # Clear text
                
                # Set current trial params
                # self.latency = self.latency_domain[0] # For now latency is constant
                # self.motion_scale = self.control_scale # Will be initialized in tk constructor and picked every end_game
                self.current_log_file = f"{self.data_folder}/trial{self.trial_num}_l{self.latency}s{self.motion_scale}.csv"
                self.target_data_file = f"{self.data_folder}/target_data_trial{self.trial_num}_l{self.latency}s{self.motion_scale}.csv"

                # Create border warning rectangles
                self.warning_rectangles = {}
                border_margin = 50
                self.warning_rectangles["left"] = self.canvas.create_rectangle(*self.border_coordinates("left"), outline="", fill="white")
                self.warning_rectangles["right"] = self.canvas.create_rectangle(*self.border_coordinates("right"), outline="", fill="white")
                self.warning_rectangles["top"] = self.canvas.create_rectangle(*self.border_coordinates("top"), outline="", fill="white")
                self.warning_rectangles["bottom"] = self.canvas.create_rectangle(*self.border_coordinates("bottom"), outline="", fill="white")
                self.border_warning = {"left": False, "right": False, "top": False, "bottom": False}

                # Generate targets
                self.generate_targets(self.target_distance, self.target_width)                
                
                # Hide mouse cursor
                self.root.config(cursor="none")
                
                # Create the instrument at the same position as the start button
                start_button_x, start_button_y = self.screen_center_x, self.screen_center_y
                self.instrument = self.canvas.create_oval(
                        start_button_x - 5, start_button_y - 5, start_button_x + 5, start_button_y + 5, fill="blue"
                )
                self.start_button.destroy()
                self.quit_button.destroy()
                self.start_practice_button.destroy()
                
                # Create clutch label, clutch on by default at start
                self.clutch_active = True  # Flag to track clutch state on master side
                self.slave_clutch_active = True # Flag to track clutch state on slave side
                self.clutch_status_label = tk.Label(root, text="Clutch: On", fg="green", font=("Arial", 20))
                self.clutch_status_label.place(x=10, y=10)
                
                # bind click and clutch events
                self.canvas.bind("<Button-1>", self.send_click_mouse)
                self.root.bind("<space>", self.send_toggle_clutch)
                self.prev_mouse_x, self.prev_mouse_y = self.root.winfo_pointerx(), self.root.winfo_pointery()      
                
                # Start tracking
                self.game_start_time = time.time()
                self.track_mouse()
                        

        # Two functions for triggering and actually calling clutch toggle to simulate latency properly
        def send_toggle_clutch(self, event):
                self.clutch_active = not self.clutch_active
                self.root.after(int(self.latency*1000), self.toggle_clutch)
                
        def toggle_clutch(self):
                if self.clutch_active:
                        self.slave_clutch_active = True
                        self.clutch_status_label.config(text="Clutch: On", fg="green", font=("Arial", 20))
                        self.clutch_status_label.place(x=10, y=10)
                        self.canvas.itemconfig(self.instrument, fill="blue")
                        self.root.config(cursor="none")
                else:
                        self.slave_clutch_active = False
                        self.clutch_status_label.config(text="Clutch: Off", fg="red", font=("Arial", 80))
                        label_width = self.clutch_status_label.winfo_reqwidth()
                        self.clutch_status_label.place(x = self.screen_center_x - (label_width // 2), y = 250)
                        self.canvas.itemconfig(self.instrument, fill="gray")
                        self.root.config(cursor="")
                        #self.canvas.tag_lower(self.clutch_status_label)

        # Generate target display
        def generate_targets(self, distance, diameter):

                initial_angle = random.uniform(0, 2 * math.pi)
                direction = random.choice([-1, 1])
                angle_increment = 2 * math.pi / self.num_targets

                for i in range(self.num_targets):
                        angle = initial_angle + direction * (i * np.pi + np.floor(i/2) * angle_increment)
                        target_x = self.screen_center_x + (distance/2) * math.cos(angle)
                        target_y = self.screen_center_y + (distance/2) * math.sin(angle)
                        shape = self.canvas.create_oval(target_x - diameter / 2, target_y - diameter / 2, target_x + diameter / 2, target_y + diameter / 2, fill="red")
                        # label = self.canvas.create_text(target_x, target_y, text=str(i + 1), fill="white") 
                        self.target_positions.append((target_x, target_y))
                        self.target_shapes.append(shape)

                self.canvas.itemconfig(self.target_shapes[0], fill="green")  # Change target color to green
                self.canvas.itemconfig(self.target_shapes[1], fill="yellow")  # Change target color to green


        
        def track_mouse(self):

                mouse_x, mouse_y = self.root.winfo_pointerx(), self.root.winfo_pointery()

                # Get instrument pos
                instrument_coords = self.canvas.coords(self.instrument)
                instrument_x = (instrument_coords[0] + instrument_coords[2]) / 2
                instrument_y = (instrument_coords[1] + instrument_coords[3]) / 2

                if self.num_clicks >= 1:
                        data_point = [time.time() - self.game_start_time] + [instrument_x, instrument_y] + [False] + [self.slave_clutch_active]
                        self.game_data.append(data_point)

                # Check if the mouse is near the window borders
                self.update_warning_rectangle(mouse_x, mouse_y)

                # Add mouse position and time stamp to the queue
                self.mouse_queue.append((mouse_x, mouse_y, time.time(), self.clutch_active))

                # Only update instrument position when latency has been reached 
                if self.mouse_queue[-1][2] - self.mouse_queue[0][2] >= self.latency:
                        cur_mouse_queue = self.mouse_queue.popleft()
                        mouse_x = cur_mouse_queue[0]
                        mouse_y = cur_mouse_queue[1]

                        # Calculate instrument movement based on mouse position and scaling
                        dx = (mouse_x - self.prev_mouse_x) * self.motion_scale
                        dy = (mouse_y - self.prev_mouse_y) * self.motion_scale
                        # Update instrument position
                        if cur_mouse_queue[3]:
                                self.canvas.move(self.instrument, dx, dy)
                        self.prev_mouse_x, self.prev_mouse_y = mouse_x, mouse_y    

                # Repeat to continuously call function
                self.after_id = self.root.after(10, self.track_mouse)

                
        # mouse click binds to send_click_mouse, which calls click_mouse after latency
        def send_click_mouse(self, event):
                if self.clutch_active:
                        self.root.after(int(self.latency*1000), self.click_mouse)
                
        def click_mouse(self):
                self.num_clicks += 1
                # print(f"click #{self.num_clicks}")
                if self.num_clicks == 1:
                        self.game_start_time = time.time()
                if self.num_clicks <= self.num_targets:
                    # Get instrument pos
                    instrument_coords = self.canvas.coords(self.instrument)
                    instrument_x = (instrument_coords[0] + instrument_coords[2]) / 2
                    instrument_y = (instrument_coords[1] + instrument_coords[3]) / 2

                    # save data point with click True
                    data_point = [time.time() - self.game_start_time] + [instrument_x, instrument_y] + [True] + [self.slave_clutch_active]
                    self.game_data.append(data_point)

                    if self.current_target < self.num_targets-2:
                            self.target_hit[self.current_target] = True
                            self.canvas.itemconfig(self.target_shapes[self.current_target], fill="red")
                            self.canvas.itemconfig(self.target_shapes[self.current_target+1], fill="green") 
                            self.canvas.itemconfig(self.target_shapes[self.current_target+2], fill="yellow") 
                            self.current_target += 1

                    elif self.current_target == self.num_targets-2:
                            self.canvas.itemconfig(self.target_shapes[self.current_target], fill="red")
                            self.canvas.itemconfig(self.target_shapes[self.current_target+1], fill="green")
                            self.current_target += 1

                    elif self.current_target == self.num_targets-1: # Last target clicked
                            if self.practice_mode:
                                    self.clear_practice_targets()
                            else:
                                    self.end_trial()

        ## Game ended after all targets clicked
        def end_trial(self):
                self.trial_running = False
                self.game_end_time = time.time()
                self.clutch_active = False
                self.clutch_status_label.config(text="Clutch: Off", fg="red")
                
                # Show the mouse cursor again
                self.root.config(cursor="")

                # Unbind buttons and stop track_mouse
                self.canvas.unbind("<Button-1>")
                self.root.unbind("<space>")
                self.root.after_cancel(self.after_id)
                
                # Create the "Save Data" and "Don't Save" buttons after the game ends
                self.save_button = tk.Button(self.root, text="Save Data", command=self.save_game_data)
                self.save_button.place(relx=0.35, rely=0.9)
                
                self.dont_save_button = tk.Button(self.root, text="Don't Save", command=self.dont_save_game_data)
                self.dont_save_button.place(relx=0.65, rely=0.9)
 
        ## Bound to save data button, saves trial data and moves onto next trial
        def save_game_data(self):
                self.save_data = True
                self.save_game_data_to_csv()
                self.visited_params['latency'].append(self.latency)
                self.visited_params['scale'].append(self.motion_scale)

                ### Model updates and scaling policy code go here
                self.feedback() # updates self.model and self.control_scale

                self.trial_num += 1
                # TO DO: Need to update conditions for ending game
                # if self.trial_num == self.total_trials:
                #         self.root.destroy()
                #         return "break"

                # Otherwise clear data and return to start
                self.clear_game_data()
                self.display_start_screen()
                
        
        def dont_save_game_data(self):

                # Clear game data without saving
                self.clear_game_data()
                
                # Return to the initial start screen
                self.display_start_screen()
        
        def clear_game_data(self):
                self.instrument = None
                self.target_positions = []
                self.target_hit = [False] * self.num_targets
                self.target_distances = [None] * self.num_targets
                self.target_shapes = []
                self.current_target = 0
                self.movement_data = []
                self.game_data = []
                self.trial_running = False
                self.game_start_time = None
                self.game_end_time = None
                self.save_data = False
                self.prev_mouse_x = None
                self.prev_mouse_y = None
                self.mouse_queue.clear()

        ### displays starting screen
        def display_start_screen(self):
                
                # Clear canvas
                self.canvas.delete("all")

                if self.save_button:
                        self.save_button.destroy()
                        self.save_button = None
                if self.dont_save_button:
                        self.dont_save_button.destroy()
                        self.dont_save_button = None

                self.start_button = tk.Button(self.root, text="Start", command=self.start_game)
                self.start_button.place(x=self.screen_center_x, y=self.screen_center_y)

                self.start_practice_button = tk.Button(self.root, text="Practice", command=self.start_practice_mode)
                self.start_practice_button.place(x=self.screen_center_x, y=self.screen_center_y + 50)

                self.quit_button = tk.Button(self.root, text="Quit", command=self.root.destroy)
                self.quit_button.place(x=self.screen_center_x, y=self.screen_center_y + 100)

                if hasattr(self, 'clutch_status_label'):
                        self.clutch_status_label.destroy()

                

                # Display instructions and trial info
                self.latency = self.latency_domain[0] # For now latency is constant
                self.motion_scale = self.control_scale # Will be initialized in tk constructor and picked every end_game
                # trials_left = self.total_trials - self.trial_num
                trial_message = (f"Trial #{self.trial_num + 1}. "
                                 f"Latency = {self.latency}, Scaling Factor = {self.motion_scale}\n"
                                #  f"Number of trials left: {trials_left}"
                                 )
                self.canvas.create_text(self.screen_center_x, 200, text=trial_message, font=("Arial", 16))
                instructions = ("Instructions:\n"
                                "- Click \'Start\' to initialize the trial. (You may want to read the read the rest of the instructions before doing so).\n"
                                "- Each trial consists of 10 targets that you will try to click as quickly as possible. The simulated latency and motion scaling factor will change for each trial.\n"
                                "- Your goal is to click the targets as quickly and accurately as possible. Green indicates the target you are currently trying to hit, and the subsequent target is indicated in yellow.\n"
                                "- Please only attempt to click each target once, as each click will automatically trigger the next target.\n"
                                "- The trial only begins when the first target is clicked (not when \'Start\' is clicked). You should therefore take your time, get used to the delay and scaling factor, before clicking the center of the first target.\n"
                                "- Mouse motion is tracked only while the clutch is active. Press spacebar to toggle the clutch on/off. You are encouraged to do a few practice trials to get used to the clutch mechanism.\n"
                                "- A red border and warning message will indicate when the mouse is close to the screen border. You should toggle the clutch and readjust when you see this warning.\n"
                                "- The trial ends automatically after the last target is clicked. Then click \'Save\' to move on to the next trial or \'Dont Save\' to redo the trial.\n"
                                )
                self.canvas.create_text(self.screen_center_x, 800, text=instructions, font=("Arial", 14))
                

                
        def save_game_data_to_csv(self):
                if self.save_data:
                        with open(self.param_file, mode="a", newline="") as file:
                                writer = csv.writer(file)
                                latency = self.latency
                                scaling_factor = self.motion_scale
                                # target_distances = self.target_distances
                                # total_time = self.game_end_time - self.game_start_time
                                data_row = [latency, scaling_factor, self.target_distance, self.target_width] #+ target_distances + [total_time]
                                writer.writerow(data_row)

                        # Save motion data
                        df = pd.DataFrame(self.game_data, columns=self.data_header)
                        df = df.sort_values(by="time")
                        df.to_csv(self.current_log_file, index=False, mode='w')

                        # Save target position data
                        df_target = pd.DataFrame(self.target_positions)
                        df_target.to_csv(self.target_data_file, mode='w')

        def display_warning_message(self):
                if not hasattr(self, "warning_message"):
                        # Create a warning label if it doesn't exist
                        self.warning_message = self.canvas.create_text(
                                self.canvas.winfo_width() / 2, (self.canvas.winfo_height() / 2)-150,
                                text="WARNING: Mouse near window border!",
                                font=("Arial", 16),
                                fill="red"
                        )
                        
        def clear_warning_message(self):
                if hasattr(self, "warning_message"):
                        # Clear the warning label if it exists
                        self.canvas.delete(self.warning_message)
                        delattr(self, "warning_message")


        def update_warning_rectangle(self, mouse_x, mouse_y):
                
                border_margin = 100  # Margin in pixels to trigger the warning
                near_left_border = mouse_x < border_margin
                near_right_border = mouse_x > (self.screen_width - border_margin)
                near_top_border = mouse_y < border_margin
                near_bottom_border = mouse_y > (self.screen_height - border_margin)

                if near_left_border:
                        self.border_warning["left"] = True
                        self.canvas.itemconfig(self.warning_rectangles["left"], fill="red")
                        self.display_warning_message()
                elif self.border_warning["left"]:
                        self.canvas.itemconfig(self.warning_rectangles["left"], fill="white")
                        self.border_warning["left"] = False
                        self.clear_warning_message()

                if near_right_border:
                        self.border_warning["right"] = True
                        self.canvas.itemconfig(self.warning_rectangles["right"], fill="red")
                        self.display_warning_message()                        
                elif self.border_warning["right"]:
                        self.canvas.itemconfig(self.warning_rectangles["right"], fill="white")
                        self.border_warning["right"] = False
                        self.clear_warning_message()

                if near_top_border:
                        self.border_warning["top"] = True
                        self.canvas.itemconfig(self.warning_rectangles["top"], fill="red")
                        self.display_warning_message()                        
                elif self.border_warning["top"]:
                        self.canvas.itemconfig(self.warning_rectangles["top"], fill="white")
                        self.border_warning["top"] = False
                        self.clear_warning_message()

                if near_bottom_border:
                        self.border_warning["bottom"] = True
                        self.canvas.itemconfig(self.warning_rectangles["bottom"], fill="red")
                        self.display_warning_message()                        
                elif self.border_warning["bottom"]:
                        self.canvas.itemconfig(self.warning_rectangles["bottom"], fill="white")
                        self.border_warning["bottom"] = False
                        self.clear_warning_message()
                        

        def border_coordinates(self, key):
                border_margin = 50
                
                if key == 'left':
                        return 0, 0, border_margin, self.screen_height
                elif key == 'right':
                        return self.screen_width - border_margin, 0, self.screen_width, self.screen_height
                elif key == 'top':
                        return 0, 0, self.screen_width, border_margin
                elif key == 'bottom':
                        return 0, self.screen_height - border_margin, self.screen_width, self.screen_height
                
                

        def toggle_fullscreen(self, event=None):
                self.fullscreen_state = not self.fullscreen_state  # Just toggling the boolean
                self.root.attributes("-fullscreen", self.fullscreen_state)
                return "break"

        def end_fullscreen(self, event=None):
                self.fullscreen_state = False
                self.root.attributes("-fullscreen", False)
                return "break"


        def start_practice_mode(self):

                # Set flags
                self.practice_mode = True
                self.num_clicks = 0

                self.canvas.delete("all") # Clear text
                
                # Create slider controls
                self.latency = 0.0
                self.motion_scale = 1.0
                self.latency_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.1, orient=tk.VERTICAL,
                                               label="Latency", command=self.update_latency)
                self.latency_slider.place(x=10, y=50)
                self.latency_slider.set(0.0)

                self.motion_scale_slider = tk.Scale(root, from_=0.05, to=1.0, resolution=0.05, orient=tk.VERTICAL,
                                                    label="Motion Scale", command=self.update_motion_scale)
                self.motion_scale_slider.place(x=10, y=200)
                self.motion_scale_slider.set(1.0)

                # Create border warning rectangles
                self.warning_rectangles = {}
                self.warning_rectangles["left"] = self.canvas.create_rectangle(*self.border_coordinates("left"), outline="", fill="white")
                self.warning_rectangles["right"] = self.canvas.create_rectangle(*self.border_coordinates("right"), outline="", fill="white")
                self.warning_rectangles["top"] = self.canvas.create_rectangle(*self.border_coordinates("top"), outline="", fill="white")
                self.warning_rectangles["bottom"] = self.canvas.create_rectangle(*self.border_coordinates("bottom"), outline="", fill="white")
                self.border_warning = {"left": False, "right": False, "top": False, "bottom": False}

                # Generate targets
                self.generate_targets(self.target_distance, self.target_width)                
                
                # Hide mouse cursor
                self.root.config(cursor="none")
                
                # Create the instrument at the same position as the start button
                start_button_x, start_button_y = self.screen_center_x, self.screen_center_y
                self.instrument = self.canvas.create_oval(
                        start_button_x - 5, start_button_y - 5, start_button_x + 5, start_button_y + 5, fill="blue"
                )
                self.start_button.destroy()
                self.quit_button.destroy()
                self.start_practice_button.destroy()
                
                # Create clutch label, clutch on by default at start
                self.clutch_active = True  # Flag to track clutch state on master side
                self.slave_clutch_active = True # Flag to track clutch state on slave side
                self.clutch_status_label = tk.Label(root, text="Clutch: On", fg="green", font=("Arial", 20))
                self.clutch_status_label.place(x=10, y=10)

                # Prompt to quit practice mode
                self.exit_practice_msg = self.canvas.create_text(self.screen_center_x, 60, text="Press \'q\' to exit practice mode", font=("Arial", 16))
                self.root.bind("<q>", self.exit_practice_mode)
                
                # bind click and clutch events
                self.canvas.bind("<Button-1>", self.send_click_mouse)
                self.root.bind("<space>", self.send_toggle_clutch)
                self.prev_mouse_x, self.prev_mouse_y = self.root.winfo_pointerx(), self.root.winfo_pointery()
                
                # Start tracking
                self.track_mouse()

        
        def update_latency(self, value):
                self.root.after_cancel(self.after_id)
                self.mouse_queue.clear()
                self.latency = float(value)
                self.prev_mouse_x, self.prev_mouse_y = self.root.winfo_pointerx(), self.root.winfo_pointery()
                self.track_mouse()
                

        def update_motion_scale(self, value):
                self.motion_scale = float(value)

        ## clear current targets, display a message, and regenerate targets        
        def clear_practice_targets(self):
                self.root.after_cancel(self.after_id)
                self.mouse_queue.clear()
                self.canvas.unbind("<Button-1>")
                self.canvas.delete(self.instrument)
                for t in self.target_shapes:
                        self.canvas.delete(t)
                msg = self.canvas.create_text(self.screen_center_x, self.screen_center_y, text="Trial Finished. Nice Job!", font=("Arial", 16))
                self.root.after(1000, self.regen_practice_targets, msg)

        def regen_practice_targets(self, msg):
                self.canvas.delete(msg)
                self.target_positions = []
                self.target_shapes = []
                self.current_target = 0
                self.num_clicks = 0
                self.generate_targets(self.target_distance, self.target_width)
                # Create the instrument at the same position as the start button
                start_button_x, start_button_y = self.screen_center_x, self.screen_center_y
                self.instrument = self.canvas.create_oval(
                        start_button_x - 5, start_button_y - 5, start_button_x + 5, start_button_y + 5, fill="blue"
                )
                self.canvas.bind("<Button-1>", self.send_click_mouse)
                self.prev_mouse_x, self.prev_mouse_y = self.root.winfo_pointerx(), self.root.winfo_pointery()
                self.track_mouse()


        def exit_practice_mode(self, event):

                self.practice_mode = False
                self.clutch_active = False
                self.clutch_status_label.destroy()
                self.latency_slider.destroy()
                self.motion_scale_slider.destroy()
                
                # Show the mouse cursor again
                self.root.config(cursor="")

                # Unbind buttons and stop track_mouse
                self.canvas.unbind("<Button-1>")
                self.root.unbind("<q>")
                self.root.unbind("<space>")
                self.root.after_cancel(self.after_id)

                self.clear_game_data()
                self.display_start_screen()


        def process_trial(self, latency, scale, trial_data_file, target_data_file):
                '''
                Calculatetes performance metrics for current trial
                args:
                returns:
                    - input: (latency, scale)
                    - metric_dict: dictionary: {peformance_metric: value}
                '''

                df = pd.read_csv(trial_data_file)
                target_df = pd.read_csv(target_data_file)

                # Find the indices where "click" is True
                click_indices = df.index[df['click']]
                df_noclick = df[~df["click"]]

                if len(click_indices) != 10:
                    print(f"Warning! Data for {latency} latency and {scale} scale has {len(click_indices)} clicks!")

                target_distances = []
                movement_distances = []
                movement_times = []
                movement_speeds = []
                target_errors = [] # deviations of select point to intended target point
                target_distances = []
                end_points = []
                osd_set = []
                t_eff_set = []

                # Calculate mean and standard deviation of sampling rate in motion data file
                dt = df_noclick["time"].diff()
                fs = 1.0 / dt
                fs_mean = np.mean(fs)
                fs_std = np.std(fs)
                if fs_std > 5:
                    print(f"Warning! Sampling Rate mean is {fs_mean}, std is {fs_std}")


                # # # Generate figure for metrics
                # # fig, axes = plt.subplots(2, 4, figsize=(24, 12))

                # Split the data into segments using the click indices
                for i in range(len(click_indices)-1):
                    
                    # Segment data by clicks 
                    start_idx = click_indices[i]
                    end_idx = click_indices[i+1]
                    segment = df.iloc[start_idx:end_idx]
                    

                    start_point = np.array([df['ins_x'][start_idx], df['ins_y'][start_idx]]) 
                    end_point = np.array([df['ins_x'][end_idx], df['ins_y'][end_idx]])
                    target_to = np.array([target_df['0'][i+1], target_df['1'][i+1]])
                    target_from = np.array([target_df['0'][i], target_df['1'][i]])
                    movement_axis = target_to - start_point # defined from motion start point to target center
                    trans_end_point = transform_2d_coordinate(end_point, movement_axis, target_to) # transform end points to target frame
                    target_distance_signal = np.linalg.norm(segment[['ins_x', 'ins_y']].values - target_to, axis=1)
                    osd = compute_osd(target_distance_signal, np.array(segment['time']))
                    travel_distance = sum(((segment['ins_x'].diff().fillna(0))**2 + (segment['ins_y'].diff().fillna(0))**2)**0.5)
                    translation_efficiency = np.linalg.norm(movement_axis) / travel_distance
                    
                    osd_set.append(osd)
                    end_points.append(trans_end_point)
                    movement_distances.append(math.dist(start_point, end_point)) # Euclidean dist between start and end points
                    movement_times.append(df['time'][end_idx] - df['time'][start_idx])
                    movement_speeds.append(movement_distances[-1] / movement_times[-1])
                    target_distances.append(math.dist(target_from, target_to))
                    target_errors.append(math.dist(end_point, target_to))
                    t_eff_set.append(translation_efficiency)

                total_osd = np.sum(osd_set)
                avg_osd = np.mean(osd_set)
                avg_movement_speed = np.mean(np.array(movement_distances) / np.array(movement_times))
                avg_target_error = np.mean(target_errors)
                total_target_error = np.sum(target_errors)
                target_error_rate = (sum(1 for error in target_errors if error > 20) / len(target_errors)) * 100
                effective_distance = np.mean(movement_distances)
                end_point_std = np.linalg.norm(np.std(np.array(end_points), axis=0)) # standard deviation of end point scatter
                effective_width = 4.133 * end_point_std
                effective_difficulty = math.log2((effective_distance / effective_width) + 1)
                # difficulty = math.log2((np.mean(target_distances) / 40) + 1)
                # print('Target Distance = ', np.mean(target_distances), 'Theoretical ID = ', difficulty)
                avg_movement_time = np.mean(movement_times)
                throughput = effective_difficulty / avg_movement_time
                avg_translation_efficiency = np.mean(t_eff_set)
                total_error = avg_osd + avg_target_error

                num_clutches = sum((df['clutch']) & (df['clutch'].shift(-1) == False))

                current_obs_data = dict(
                        latency = latency,
                        scale = scale,
                        throughput = throughput,
                        avg_osd = avg_osd,
                        avg_target_error = avg_target_error,
                        total_error = total_error,
                        avg_movement_speed = avg_movement_speed
                )

                self.obs_data.append(current_obs_data)

                return current_obs_data

        def feedback(self):
                # Process trial data, update model
                current_obs_data = self.process_trial(self.latency, self.motion_scale, self.current_log_file, self.target_data_file)
                train_input = np.array([current_obs_data['latency'], current_obs_data['scale']]).reshape((-1, 1))
                train_output_dict = {key: value for key, value in current_obs_data.items() if key not in ["latency", "scale"]}
                self.model.add_training_data(train_input, train_output_dict)
                self.model.train()
                prediction_dict = self.model.predict(self.prediction_input, self.prediction_df)

                # Choosing next scaling factor
                # if len(self.visited_params['scale']) < 10:
                #         self.control_scale = self.scaling_policy.random_scale(self.visited_params['scale'])
                # else:
                #         print("Went through all scale domain")
                #         self.control_scale = 1.0
                self.control_scale = self.scaling_policy.get_scale(self.prediction_df, metric="throughput")

                # Visualize results
                self.visualize_feedback()


        def visualize_feedback(self):

                save_folder = os.path.join(self.data_folder, "controller_results")
                os.makedirs(save_folder, exist_ok=True)

                # Create DataFrame for all combinations
                sparse_df = pd.DataFrame(list(itertools.product(self.latency_domain, self.scale_domain)), columns=["latency", "scale"])
                sparse_df['throughput'] = np.nan
                sparse_df['total_error'] = np.nan

                # Update sparse_df with data from obs_df
                # header = ['latency', 'scale', 'throughput', 'avg_osd', 'avg_target_error', 'total_error', 'avg_movement_speed']
                obs_df_copy = pd.DataFrame(self.obs_data)
                obs_df_copy = obs_df_copy.groupby(['latency', 'scale']).mean().reset_index()
                sparse_df.set_index(['latency', 'scale'], inplace=True)
                obs_df_copy.set_index(['latency', 'scale'], inplace=True)
                sparse_df.update(obs_df_copy)
                sparse_df.reset_index(inplace=True)

                # Plotting
                fig, axes = plt.subplots(3, 2, figsize=(12, 6))
                fig.suptitle(f"Control Scale chosen: {self.control_scale}")

                # Throughput Heatmap
                sparse_throughput_heatmap = sparse_df.pivot(index="latency", columns="scale", values="throughput")
                sns.heatmap(sparse_throughput_heatmap, cmap='YlGnBu', ax=axes[0, 0], annot=True)
                axes[0, 0].set_title('Observed Throughput')
                axes[0, 0].set_xlabel('Scale')
                axes[0, 0].set_ylabel('Latency')

                # Total Error Heatmap
                sparse_error_heatmap = sparse_df.pivot(index="latency", columns="scale", values="total_error")
                sns.heatmap(sparse_error_heatmap, cmap='YlGnBu', ax=axes[0, 1], annot=True)
                axes[0, 1].set_title('Observed Total Error')
                axes[0, 1].set_xlabel('Scale')
                axes[0, 1].set_ylabel('Latency')

                # Predicted Heatmaps
                pred_throughput_heatmap = self.prediction_df.pivot(index="latency", columns="scale", values="throughput")
                sns.heatmap(pred_throughput_heatmap, cmap='YlGnBu', ax=axes[1, 0], annot=True)
                axes[1, 0].set_title('Predicted Mean Throughput')
                axes[1, 0].set_xlabel('Scale')
                axes[1, 0].set_ylabel('Latency')

                # Total Error Heatmap
                pred_error_heatmap = self.prediction_df.pivot(index="latency", columns="scale", values="total_error")
                sns.heatmap(pred_error_heatmap, cmap='YlGnBu', ax=axes[1, 1], annot=True)
                axes[1, 1].set_title('Predicted Mean Total Error')
                axes[1, 1].set_xlabel('Scale')
                axes[1, 1].set_ylabel('Latency')

                # Predicted Heatmaps
                pred_throughput_covar_heatmap = self.prediction_df.pivot(index="latency", columns="scale", values="throughput_var")
                sns.heatmap(pred_throughput_covar_heatmap, cmap='YlGnBu', ax=axes[2, 0], annot=True)
                axes[2, 0].set_title('Predicted Variance Throughput')
                axes[2, 0].set_xlabel('Scale')
                axes[2, 0].set_ylabel('Latency')

                # Total Error Heatmap
                pred_error_covar_heatmap = self.prediction_df.pivot(index="latency", columns="scale", values="total_error_var")
                sns.heatmap(pred_error_covar_heatmap, cmap='YlGnBu', ax=axes[2, 1], annot=True)
                axes[2, 1].set_title('Predicted Variance Total Error')
                axes[2, 1].set_xlabel('Scale')
                axes[2, 1].set_ylabel('Latency')


                plt.tight_layout()
                plt.savefig(os.path.join(save_folder, f"{self.trial_num}.png"))
                plt.close()
                # plt.show()
                
                


if __name__ == "__main__":

        # Input user, create data folder if needed
        user_name = input("Please type your name and then press enter: ")
        data_folder = f"controller_data_files/user_{user_name}"
        os.makedirs(data_folder, exist_ok=True)
        # if not os.path.exists(data_folder):
        #         print("creating data folder")
        #         os.mkdir(data_folder)
        

        root = tk.Tk()  
        app = InstrumentTracker(root, data_folder)
        root.mainloop()

        print("\nThanks for playing!! :D")
                
                
