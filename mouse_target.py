import os
from collections import deque
import tkinter as tk
import time
import math
import random
import csv
import pandas as pd
import numpy as np

class InstrumentTracker:
        def __init__(self, root, params, data_folder):
                self.root = root
                self.root.title("Instrument Tracker")
                
                # Set the window size to match the screen's dimensions
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                self.root.geometry(f"{screen_width}x{screen_height}")
                
                self.canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white")
                self.canvas.pack()

                # Start button
                self.start_button = tk.Button(root, text="Start", command=self.start_game)
                self.start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                
                self.save_button = None  # Button to save data
                self.dont_save_button = None  # Button to return to the initial screen without saving
                self.save_data = False  # Flag to determine whether to save data
                
                self.instrument = None
                
                self.target_positions = []
                self.num_targets = 10
                self.target_hit = [False] * self.num_targets  # Track target hits
                self.target_shapes = []  # Store target shapes
                self.current_target = 0  # Track the current target
                self.target_distance = 200
                self.target_width = 20

                self.trial_running = False
                self.num_clicks = 0
                
                self.mouse_data = deque()
                self.game_data = []
                self.movement_data = []
                self.game_running = False
                self.game_start_time = None  # Timestamp when the game started
                self.game_end_time = None  # Timestamp when the game ended

                self.data_header = ["time", "ins_x", "ins_y", "click"]

                # set params
                self.latency = params[0]
                self.motion_scale = params[1]

                # set logging files
                self.param_file = f"{data_folder}/tested_params.csv"
                self.current_log_file = f"{data_folder}/l{self.latency}s{self.motion_scale}.csv"
                self.target_data_file = f"{data_folder}/target_data_l{self.latency}s{self.motion_scale}.csv"

                
        
        def start_game(self):
                
                # Destroy the "Save Data" and "Don't Save" buttons if they exist
                if self.save_button:
                        self.save_button.destroy()
                        self.save_button = None
                if self.dont_save_button:
                        self.dont_save_button.destroy()
                        self.dont_save_button = None

                # Create clutch label, clutch on by default at start
                self.clutch_active = True  # Flag to track clutch state
                self.clutch_status_label = tk.Label(root, text="Clutch: On", fg="green", font=("Arial", 16))
                self.clutch_status_label.place(x=10, y=10)

                # Generate targets
                self.generate_targets(self.target_distance, self.target_width)
                
                self.save_data = False
                self.game_running = True
                self.first_click = False

                # Hide mouse cursor
                self.root.config(cursor="none")
                
                # Create the instrument at the same position as the start button
                start_button_x, start_button_y = self.start_button.winfo_x(), self.start_button.winfo_y()
                self.instrument = self.canvas.create_oval(
                        start_button_x - 5, start_button_y - 5, start_button_x + 5, start_button_y + 5, fill="blue"
                )
                self.start_button.destroy()  # Remove the start button

                
                # bind click and clutch events
                self.canvas.bind("<Button-1>", self.send_click_mouse)
                self.root.bind("<space>", self.send_toggle_clutch)
                self.prev_mouse_x, self.prev_mouse_y = self.root.winfo_pointerx(), self.root.winfo_pointery()      

                # Turn clutch on
                self.clutch_active = True
                self.clutch_status_label.config(text="Clutch: On", fg="green")
                
                # Start tracking
                self.game_start_time = time.time()
                self.track_mouse()
                        

        # Two functions for triggering and actually calling clutch toggle to simulate latency properly
        def send_toggle_clutch(self, event):
                self.clutch_active = not self.clutch_active
                self.root.after(int(self.latency*1000), self.toggle_clutch)
                
        def toggle_clutch(self):
                if self.clutch_active:
                        self.clutch_status_label.config(text="Clutch: On", fg="green")
                        self.root.config(cursor="none")
                else:
                        self.clutch_status_label.config(text="Clutch: Off", fg="red")
                        self.root.config(cursor="")

        # Generate target display
        def generate_targets(self, distance, diameter):

                initial_angle = random.uniform(0, 2 * math.pi)
                direction = random.choice([-1, 1])
                angle_increment = 2 * math.pi / self.num_targets
                
                window_center_x, window_center_y = self.start_button.winfo_x(), self.start_button.winfo_y()

                for i in range(self.num_targets):
                        angle = initial_angle + direction * (i * np.pi + np.floor(i/2) * angle_increment)
                        target_x = window_center_x + distance * math.cos(angle)
                        target_y = window_center_y + distance * math.sin(angle)
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
                        data_point = [time.time() - self.game_start_time] + [instrument_x, instrument_y] + [False]
                        self.game_data.append(data_point)

                # Check if the mouse is near the window borders
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                border_margin = 20  # Margin in pixels to trigger the warning

                near_left_border = mouse_x < 5*border_margin
                near_right_border = mouse_x > (screen_width - border_margin)
                near_top_border = mouse_y < 5*border_margin
                near_bottom_border = mouse_y > (screen_height - border_margin)

                if near_left_border or near_right_border or near_top_border or near_bottom_border:
                        self.display_warning_message()
                else:
                        self.clear_warning_message()

                # Add mouse position and time stamp to the queue
                self.mouse_data.append((mouse_x, mouse_y, time.time(), self.clutch_active))

                # Only update instrument position when latency has been reached 
                if self.mouse_data[-1][2] - self.mouse_data[0][2] >= self.latency:
                        cur_mouse_data = self.mouse_data.popleft()
                        mouse_x = cur_mouse_data[0]
                        mouse_y = cur_mouse_data[1]

                        # Calculate instrument movement based on mouse position and scaling
                        dx = (mouse_x - self.prev_mouse_x) * self.motion_scale
                        dy = (mouse_y - self.prev_mouse_y) * self.motion_scale
                        # Update instrument position
                        if cur_mouse_data[3]:
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
                if self.num_clicks == 1:
                        self.game_start_time = time.time()
                
                # Get instrument pos
                instrument_coords = self.canvas.coords(self.instrument)
                instrument_x = (instrument_coords[0] + instrument_coords[2]) / 2
                instrument_y = (instrument_coords[1] + instrument_coords[3]) / 2

                # save data point with click True
                data_point = [time.time() - self.game_start_time] + [instrument_x, instrument_y] + [True]
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

                elif self.current_target == self.num_targets-1:
                        self.end_game()

        # Game ended after all targets clicked
        def end_game(self):
                self.game_running = False
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
                self.save_button.place(relx=0.35, rely=0.9, anchor=tk.CENTER)
                
                self.dont_save_button = tk.Button(self.root, text="Don't Save", command=self.dont_save_game_data)
                self.dont_save_button.place(relx=0.65, rely=0.9, anchor=tk.CENTER)
        
        def save_game_data(self):
                # Record game data to a CSV file and quit if the "Save Data" button is clicked
                self.save_data = True
                self.save_game_data_to_csv()
                self.root.destroy()
                
        
        def dont_save_game_data(self):
                self.root.destroy()
                # # Destroy the "Save Data" and "Don't Save" buttons without saving
                # if self.save_button:
                #         self.save_button.destroy()
                #         self.save_button = None
                # if self.dont_save_button:
                #         self.dont_save_button.destroy()
                #         self.dont_save_button = None
                
                # # Return to the initial start screen
                # self.start_button = tk.Button(self.root, text="Start", command=self.start_game)
                # self.start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        def clear_game_data(self):
                self.instrument = None
                self.target_positions = []
                self.target_hit = [False] * 4
                self.target_distances = [None] * 4
                self.target_shapes = []
                self.current_target = 0
                self.movement_data = []
                self.game_running = False
                self.game_start_time = None
                self.game_end_time = None
                self.save_data = False
                self.prev_mouse_x = None
                self.prev_mouse_y = None
                self.mouse_data.clear()
                
                # Clear canvas
                self.canvas.delete("all")

                
        def save_game_data_to_csv(self):
                if self.save_data:
                        with open(self.param_file, mode="a", newline="") as file:
                                writer = csv.writer(file)
                                latency = self.latency
                                scaling_factor = self.motion_scale
                                # target_distances = self.target_distances
                                # total_time = self.game_end_time - self.game_start_time
                                data_row = [latency, scaling_factor] #+ target_distances + [total_time]
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
                                self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                                text="WARNING: Mouse near window border!",
                                font=("Arial", 16),
                                fill="red"
                        )
                        
        def clear_warning_message(self):
                if hasattr(self, "warning_message"):
                        # Clear the warning label if it exists
                        self.canvas.delete(self.warning_message)
                        delattr(self, "warning_message")

if __name__ == "__main__":

        # Data log folder
        set_num = 2
        data_folder = f"data_files/set{set_num}"
        if not os.path.exists(data_folder):
                os.mkdir(data_folder)
        
        # generate and shuffle parameter combinations
        latencies = [round(0.15 * i, 2) for i in range(6)]
        scales = [round(0.2 * j + 0.2, 1) for j in range(5)]
        print(latencies, scales)

        # Create a list of all possible combinations of (x, y)
        game_params = [(x, y) for x in latencies for y in scales]

        # Read already performed params from data file
        param_file = f"{data_folder}/tested_params.csv"
        data_to_remove = []

        if os.path.exists(param_file):
                with open(param_file, 'r') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        for row in csv_reader:
                                if len(row) >= 2:  # Ensure each row has at least 2 entries
                                        x_value = float(row[0])
                                        y_value = float(row[1])
                                        data_to_remove.append((x_value, y_value))

                                        # Remove the extracted combinations from the list of tuples
                                        for item in data_to_remove:
                                                if item in game_params:
                                                        game_params.remove(item)

        # Randomize the order of the tuples
        random.shuffle(game_params)

        total_trials = len(game_params)
        print("Total trials left: ", total_trials)

        
        for i, params in enumerate(game_params):
                
                print('Game #', i, " Params: ", params[0], params[1])
                root = tk.Tk()  
                app = InstrumentTracker(root, params, data_folder)
                root.mainloop()

                # Check for user input to continue or exit
                user_input = input("Press Enter to continue to the next game or 'q' + Enter to quit: ")
                if user_input.lower() == 'q':
                        break
                
