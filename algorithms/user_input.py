import os
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk

def prompt_continue_or_exit():
    while True:
        choice = input("(y/n)").lower()
        if choice == 'y':
            return
        elif choice == 'n':
            sys.exit("Exiting the script...")
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def ask_plot_settings():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    max_freq = simpledialog.askfloat("Input", "Please enter max frequency:", initialvalue=50)
    y_axis_choice = simpledialog.askstring("Y-axis Setting", "Do you want an auto y-axis max for each plot or the same y-axis max for all plots?\nChoose 'auto' or 'same':", initialvalue="auto")

    return max_freq, y_axis_choice

def get_user_inputs(prompts, initialvalues, datatypes):
    root = tk.Tk()
    root.withdraw()
    dialog = MultiInputDialog(root, prompts, initialvalues, datatypes)
    # No need for root.mainloop() since the Dialog class already manages this.
    root.destroy()  # Ensure the root window is destroyed after the dialog is closed.
    return dialog.results
    
def get_directory_path(message=None):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    if not message:
        message="Select a Directory"
    directory_path = filedialog.askdirectory(title=message)

    return directory_path

def ask_plot_subtitle(default_name):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    subtitle = simpledialog.askstring("Subtitle", "Please enter a subtitle for the plot:", initialvalue=default_name)
    return subtitle

def get_directory_paths():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    directories = []
    subtitles = []
    
    while True:
        directory_path = filedialog.askdirectory(title="Select a Directory (cancel when done)")
        if not directory_path:  # Break the loop when the user cancels the dialog
            break
        folder_name = os.path.basename(directory_path)
        subtitle = ask_plot_subtitle(folder_name)
        
        directories.append(directory_path)
        subtitles.append(subtitle)

    return directories, subtitles

def get_file_path(message=None):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    if not message:
        message="Select a File"
    
    file_path = filedialog.askopenfilename(title=message)

    return file_path

def ask_user_input(prompt, initialvalue, datatype):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    if datatype == "float":
        return simpledialog.askfloat("Input", prompt, initialvalue=initialvalue)
    elif datatype == "string":
        return simpledialog.askstring("Input", prompt, initialvalue=initialvalue)

class MultiInputDialog(simpledialog.Dialog):
    def __init__(self, parent, prompts, initialvalues, datatypes):
        self.entries = []
        self.datatypes = datatypes
        self.results = []
        self.prompts = prompts
        self.initialvalues = initialvalues
        super().__init__(parent)

    def body(self, master):
        for i, prompt in enumerate(self.prompts):
            ttk.Label(master, text=prompt).grid(row=i, column=0)
            entry = ttk.Entry(master)
            entry.insert(0, str(self.initialvalues[i]))
            entry.grid(row=i, column=1)
            self.entries.append(entry)
        return self.entries[0]  # set initial focus

    def apply(self):
        for i, entry in enumerate(self.entries):
            if self.datatypes[i] == "float":
                self.results.append(float(entry.get()))
            else:
                self.results.append(entry.get())

