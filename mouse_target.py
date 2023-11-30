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
        def __init__(self, root, data_folder):
                self.root = root
                self.root.title("Instrument Tracker")

                # set logging files
                self.data_folder = data_folder
                self.param_file = f"{data_folder}/tested_params.csv"

                # Setting up game parameters
                latencies = [round(0.25 * i, 2) for i in range(4)]
                scales = [0.1, 0.15, 0.2, 0.4, 0.7, 1.0] #[round(0.2 * j + 0.2, 1) for j in range(5)]
                self.target_distance = 222
                self.target_width = 40

                # Create a list of all possible combinations of (x, y)
                self.game_params = [(x, y) for x in latencies for y in scales]

                # Removing unnecessary param combos
                self.game_params.remove((0.0, 0.1))
                # # self.game_params.remove((0.25, 0.1))
                # # self.game_params.remove((0.5, 1.0))
                self.game_params.remove((0.75, 1.0))

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
                                                        if item in self.game_params:
                                                                self.game_params.remove(item)

                # Randomize the order of the tuples, keeping (0.5, 0.1)  as first trial for training
                if (0.5, 0.1) in self.game_params:
                        self.game_params.remove((0.5, 0.1))
                random.shuffle(self.game_params)
                self.game_params.insert(0, (0.5, 0.1))
                # print(self.game_params) 

                self.total_trials = len(self.game_params)
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
                self.latency = self.game_params[self.trial_num][0]
                self.motion_scale = self.game_params[self.trial_num][1]
                self.current_log_file = f"{self.data_folder}/l{self.latency}s{self.motion_scale}.csv"
                self.target_data_file = f"{self.data_folder}/target_data_l{self.latency}s{self.motion_scale}.csv"

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
                        self.clutch_status_label.place(x = 1000, y = 200)
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
                                    self.regen_practice_targets()
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
                self.trial_num += 1
                # if completed all trials, quit application
                if self.trial_num == self.total_trials:
                        self.root.destroy()
                        return "break"

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

                

                # Display instructions and trial info
                self.latency = self.game_params[self.trial_num][0]
                self.motion_scale = self.game_params[self.trial_num][1]
                trials_left = self.total_trials - self.trial_num
                trial_message = (f"Trial #{self.trial_num + 1}. "
                                 f"Latency = {self.latency}, Scaling Factor = {self.motion_scale}\n"
                                 f"Number of trials left: {trials_left}")
                self.canvas.create_text(self.screen_center_x, 200, text=trial_message, font=("Arial", 16))
                instructions = ("Instructions:\n"
                                "- Click \'Start\' to initialize the trial. (You may want to read the read the rest of the instructions before doing so).\n"
                                "- Your goal is to click the green targets as quickly and accurately as possible. The target you will move to next is indicated in yellow.\n"
                                "- Please only attempt to click each target once, as each click will automatically trigger the next target.\n"
                                "- The trial only begins when the first target is clicked (not when \'Start\' is clicked). You should therefore take your time, get used to the delay and scaling factor, before clicking the center of the first target.\n"
                                "- Mouse motion is tracked only while the clutch is active. Press spacebar to toggle the clutch on/off. You are encouraged to do a few practice trials to get used to the clutch mechanism.\n"
                                "- A red border and warning message will indicate when the mouse is close to the screen border. You should toggle the clutch and readjust when you see this warning.\n"
                                "- The trial ends automatically after the last target is clicked. Then click \'Save\' to move on to the next trial or \'Dont Save\' to redo the trial.\n"
                                )
                self.canvas.create_text(self.screen_center_x, 800, text=instructions, font=("Arial", 16))
                

                
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
                
                # (replace this with slider controls)
                self.latency = 0.0
                self.motion_scale = 1.0

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

                # Button to exit practice mode
                self.exit_practice_button = tk.Button(self.root, text="Exit Practice Mode", command=self.exit_practice_mode)
                self.exit_practice_button.place(relx=0.9, rely=0.9)
                
                # bind click and clutch events
                self.canvas.bind("<Button-1>", self.send_click_mouse)
                self.root.bind("<space>", self.send_toggle_clutch)
                self.prev_mouse_x, self.prev_mouse_y = self.root.winfo_pointerx(), self.root.winfo_pointery()      
                
                # Start tracking
                self.track_mouse()

        

if __name__ == "__main__":

        # Input user, create data folder if needed
        user_name = input("Please type your name and then press enter: ")
        data_folder = f"data_files/user_{user_name}"
        if not os.path.exists(data_folder):
                os.mkdir(data_folder)
        

        root = tk.Tk()  
        app = InstrumentTracker(root, data_folder)
        root.mainloop()
                
        
        # for i, params in enumerate(game_params):
 
        #         print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        #         print(f'Beginning Trial #{i+1}: Latency = {params[0]}, Scaling Factor = {params[1]}')
        #         print(f'Total trials left: {total_trials-i}')
        #         # Check for user input to continue or exit
        #         user_input = input("Press Enter to continue or 'q' + Enter to quit: ")
        #         if user_input.lower() == 'q':
        #                 break
                
        #         root = tk.Tk()  
        #         app = InstrumentTracker(root, list(params) + [target_distance, target_width], data_folder)
        #         root.mainloop()

        print("\nThanks for playing!! :D")
                
                
