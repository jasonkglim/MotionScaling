from collections import deque
import tkinter as tk
import time
import math
import random
import csv

class InstrumentTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Instrument Tracker")
        
        # Set the window size to match the screen's dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        
        # Define the Task Space dimensions
        self.task_space_width = 600
        self.task_space_height = 400
        self.task_space_x = (screen_width - self.task_space_width) // 2
        self.task_space_y = (screen_height - self.task_space_height) // 2
        
        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white")
        self.canvas.pack()

        # Start button
        self.start_button = tk.Button(root, text="Start", command=self.start_game)
        self.start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.save_button = None  # Button to save data
        self.dont_save_button = None  # Button to return to the initial screen without saving
        self.save_data = False  # Flag to determine whether to save data
        
        self.instrument = None
        self.targets = []
        self.target_hit = [False] * 4  # Track target hits
        self.target_distances = [None] * 4  # Distances from instrument to targets
        self.target_shapes = []  # Store target shapes
        self.current_target = 0  # Track the current target
        self.mouse_data = deque()
        
        self.movement_data = []
        self.game_running = False
        self.game_start_time = None  # Timestamp when the game started
        self.game_end_time = None  # Timestamp when the game ended
        
        self.clutch_active = True  # Flag to track clutch state
        self.clutch_status_label = tk.Label(root, text="Clutch: On", fg="green", font=("Arial", 16))
        self.clutch_status_label.place(x=10, y=10)

        self.trial_count = 0 # number of games played so far

        # generate and shuffle parameter combinations
        latencies = [round(0.1 * i, 1) for i in range(10)]
        scales = [round(0.1 * j + 0.1, 1) for j in range(10)]

        # Create a list of all possible combinations of (x, y)
        self.game_params = [(x, y) for x in latencies for y in scales]
        # self.game_params = [(0, 1.0), (1.0, 1.0), (1.0, 0.2), (0, 0.2)]

        # # Read data from the CSV file
        # csv_filename = "game_data.csv"  # Replace with the actual CSV file name
        # data_to_remove = []

        # with open(csv_filename, 'r') as csv_file:
        #     csv_reader = csv.reader(csv_file)
        #     for row in csv_reader:
        #         if len(row) >= 2:  # Ensure each row has at least 2 entries
        #             x_value = float(row[0])
        #             y_value = float(row[1])
        #             data_to_remove.append((x_value, y_value))

        #             # Remove the extracted combinations from the list of tuples
        #             for item in data_to_remove:
        #                 if item in self.game_params:
        #                     self.game_params.remove(item)

        # Randomize the order of the tuples
        random.shuffle(self.game_params)
        

        self.total_trials = len(self.game_params)
        # print(self.total_trials)
    
    def start_game(self):
        self.clear_game_data()  # Clear previous game data
        print('Game #', self.trial_count)
        
        # Destroy the "Save Data" and "Don't Save" buttons if they exist
        if self.save_button:
            self.save_button.destroy()
            self.save_button = None
        if self.dont_save_button:
            self.dont_save_button.destroy()
            self.dont_save_button = None

        # Draw task space
        self.canvas.create_rectangle(
            self.task_space_x, self.task_space_y, self.task_space_x + self.task_space_width, self.task_space_y + self.task_space_height,
            outline="black", dash=(2, 2)
        )
        
        # Random initialization of game parameters
        self.latency = 0 #self.game_params[self.trial_count][0] #random.uniform(0.1, 0.2)
        self.motion_scale = 0.5 #self.game_params[self.trial_count][1] #random.uniform(0.9, 1.0)
        self.generate_targets()
        
        self.save_data = False
        self.game_running = True
        self.game_start_time = time.time()

        # Hide mouse cursor
        #self.root.config(cursor="none")
        
        # Create the instrument at the same position as the start button
        start_button_x, start_button_y = self.start_button.winfo_x(), self.start_button.winfo_y()
        self.instrument = self.canvas.create_oval(
            start_button_x - 5, start_button_y - 5, start_button_x + 5, start_button_y + 5, fill="blue"
        )
        self.start_button.destroy()  # Remove the start button

        
        # Start mouse tracking and click events only when the game starts
        self.canvas.bind("<Button-1>", self.click_mouse)
        self.prev_mouse_x, self.prev_mouse_y = self.root.winfo_pointerx(), self.root.winfo_pointery()
        self.root.after(1, self.track_mouse)        

        
        # Bind spacebar to toggle clutch
        self.root.bind("<space>", self.toggle_clutch)

        # Turn clutch on
        self.clutch_active = True
        self.clutch_status_label.config(text="Clutch: On", fg="green")
        
    
    def toggle_clutch(self, event):
        self.clutch_active = not self.clutch_active
        if self.clutch_active:
            self.clutch_status_label.config(text="Clutch: On", fg="green")
            # self.root.config(cursor="none")
        else:
            self.clutch_status_label.config(text="Clutch: Off", fg="red")
            self.root.config(cursor="")

    
    def generate_targets(self):
        target_positions = random.sample(range(1, 17), 4)
    
        for i, pos in enumerate(target_positions):
            row, col = divmod(pos - 1, 4)
            target_x = self.task_space_x + random.uniform(0, 1) * self.task_space_width
            target_y = self.task_space_y + random.uniform(0, 1) * self.task_space_height
            shape = self.canvas.create_oval(target_x - 15, target_y - 15, target_x + 15, target_y + 15, fill="red")
            label = self.canvas.create_text(target_x, target_y, text=str(i + 1), fill="white")
            self.targets.append((target_x, target_y))
            self.target_shapes.append(shape)


    
    def track_mouse(self):
        if self.game_running:# and self.clutch_active:
            mouse_x, mouse_y = self.root.winfo_pointerx(), self.root.winfo_pointery()

            # Check if the mouse is near the window borders
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            border_margin = 20  # Margin in pixels to trigger the warning
        
            near_left_border = mouse_x < border_margin
            near_right_border = mouse_x > (screen_width - 5*border_margin)
            near_top_border = mouse_y < border_margin
            near_bottom_border = mouse_y > (screen_height - 5*border_margin)
        
            if near_left_border or near_right_border or near_top_border or near_bottom_border:
                self.display_warning_message()
            else:
                self.clear_warning_message()

            # Add mouse position and time stamp to the queue
            self.mouse_data.append((mouse_x, mouse_y, time.time()))

            # Only update instrument position when latency has been reached 
            if self.mouse_data[-1][2] - self.mouse_data[0][2] >= self.latency:
                cur_mouse_data = self.mouse_data.popleft()
                mouse_x = cur_mouse_data[0]
                mouse_y = cur_mouse_data[1]

                # if self.mouse_data:
                #     self.prev_mouse_x = self.mouse_data[0][0]
                #     self.prev_mouse_y = self.mouse_data[0][1]
                
                # Calculate instrument movement based on mouse position and scaling
                dx = (mouse_x - self.prev_mouse_x) * self.motion_scale
                dy = (mouse_y - self.prev_mouse_y) * self.motion_scale
                # Update instrument position
                if self.clutch_active:
                    self.canvas.move(self.instrument, dx, dy)
                self.prev_mouse_x, self.prev_mouse_y = mouse_x, mouse_y
                
            
            # instrument_coords = self.canvas.coords(self.instrument)
            # instrument_x = (instrument_coords[0] + instrument_coords[2]) / 2
            # instrument_y = (instrument_coords[1] + instrument_coords[3]) / 2
            # self.canvas.move(self.instrument, dx, dy)
            
            # self.movement_data.append((time.time(), instrument_x, instrument_y))
        self.root.after(1, self.track_mouse)
                
    def click_mouse(self, event):
        if self.game_running:
            if self.current_target < 4:
                instrument_coords = self.canvas.coords(self.instrument)
                instrument_x = (instrument_coords[0] + instrument_coords[2]) / 2
                instrument_y = (instrument_coords[1] + instrument_coords[3]) / 2
                target_x, target_y = self.targets[self.current_target]
                distance = math.sqrt((target_x - instrument_x) ** 2 + (target_y - instrument_y) ** 2)
                self.target_hit[self.current_target] = True
                self.target_distances[self.current_target] = distance
                self.canvas.itemconfig(self.target_shapes[self.current_target], fill="green")  # Change target color to green
                self.current_target += 1
            
            if self.current_target == 4:
                self.end_game()
    
    def end_game(self):
        self.game_running = False
        self.game_end_time = time.time()
        self.clutch_active = False
        self.clutch_status_label.config(text="Clutch: Off", fg="red")
        
        # Show the mouse cursor again
        self.root.config(cursor="")
        
        # Create the "Save Data" and "Don't Save" buttons after the game ends
        self.save_button = tk.Button(self.root, text="Save Data", command=self.save_game_data)
        self.save_button.place(relx=0.35, rely=0.9, anchor=tk.CENTER)
        
        self.dont_save_button = tk.Button(self.root, text="Don't Save", command=self.dont_save_game_data)
        self.dont_save_button.place(relx=0.65, rely=0.9, anchor=tk.CENTER)
    
    def save_game_data(self):
        # Record game data to a CSV file if the "Save Data" button is clicked
        self.save_data = True
        self.save_game_data_to_csv()
        self.trial_count += 1
        if self.trial_count == self.total_trials:
            # end program
            self.root.quit()
        
        # Destroy the "Save Data" and "Don't Save" buttons after saving
        if self.save_button:
            self.save_button.destroy()
            self.save_button = None
        if self.dont_save_button:
            self.dont_save_button.destroy()
            self.dont_save_button = None
        
        # Return to the initial start screen
        self.start_button = tk.Button(self.root, text="Start", command=self.start_game)
        self.start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def dont_save_game_data(self):
        # Destroy the "Save Data" and "Don't Save" buttons without saving
        if self.save_button:
            self.save_button.destroy()
            self.save_button = None
        if self.dont_save_button:
            self.dont_save_button.destroy()
            self.dont_save_button = None
        
        # Return to the initial start screen
        self.start_button = tk.Button(self.root, text="Start", command=self.start_game)
        self.start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def clear_game_data(self):
        self.instrument = None
        self.targets = []
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
        
        # Clear canvas
        self.canvas.delete("all")
    
    def save_game_data_to_csv(self):
        if self.save_data:
            csv_filename = "game_data.csv"
            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                latency = self.latency
                scaling_factor = self.motion_scale
                target_distances = self.target_distances
                total_time = self.game_end_time - self.game_start_time
                data_row = [latency, scaling_factor] + target_distances + [total_time]
                writer.writerow(data_row)

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
    root = tk.Tk()
    app = InstrumentTracker(root)
    root.mainloop()
