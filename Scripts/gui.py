import configparser
import threading
import tkinter as tk
from tkinter import messagebox
import subprocess
import helper_functions as hf
from tkinter import filedialog


class App:
    """
    This class represents a GUI application for performing several tasks.

    Attributes:
        root (Tk): The main Tkinter root window.
        text_color (str): Text color for the GUI.
        button_bg (str): Background color for buttons.
        button_fg (str): Foreground color for buttons.
        entry_bg (str): Background color for input entries.
        entry_fg (str): Foreground color for input entries.
        label_fg (str): Foreground color for labels.

    Methods:
        __init__(self, root): Initializes the CryptoNewsApp instance.
        load_config(self): Loads configuration from the 'config.ini' file.
        get_script_path(self, script_name): Retrieves the script path from the configuration.
        run_script(self, script_name, output_entry): Runs a script and displays output in the terminal.
        browse_file(self, entry): Opens a file dialog to select a JSON file and updates the entry.
        run_news_crawler(self): Runs the news crawler script.
        run_sentiment_analysis(self): Runs the sentiment analysis script.
        run_sentiment_processing(self): Runs the sentiment processing script.
        add_to_terminal(self, text): Adds text to the terminal Text widget.
    """

    def __init__(self, root):
        """
        Initializes the App instance.

        Args:
            root (Tk): The main Tkinter root window.
        """
        hf.change_working_directory()

        self.root = root
        self.root.title("App")
        self.root.configure(bg="#333333")  # Dark background color

        self.load_config()

        self.TEST_MODE = self.config['Settings']['testmode'] == 'True'

        # Customize your dark theme colors
        self.text_color = "white"
        self.button_bg = "#444444"
        self.button_fg = "white"
        self.entry_bg = "#555555"
        self.entry_fg = "white"
        self.label_fg = "white"

        # Main Frame
        self.button_frame = tk.Frame(self.root, bg="#333333")
        self.button_frame.pack(pady=10)

        # News Crawler

        # Frame
        self.news_crawler_frame = tk.Frame(self.button_frame, bg="#333333")
        self.news_crawler_frame.pack(side=tk.LEFT, pady=10)

        # # No Input
        # self.input_file_label_crawler = tk.Label(
        #     self.news_crawler_frame, text="No Input", fg=self.label_fg, bg="#333333")
        # self.input_file_label_crawler.pack(side=tk.TOP, padx=10)

        # # No Input
        # self.input_file_label_crawler = tk.Label(
        #     self.news_crawler_frame, text="", fg=self.label_fg, bg="#333333")
        # self.input_file_label_crawler.pack(side=tk.TOP, padx=10)

        # # Output File Label
        # self.output_file_label_crawler = tk.Label(
        #     self.news_crawler_frame, text="Output JSON File (News Crawler):", fg=self.label_fg, bg="#333333")
        # self.output_file_label_crawler.pack(side=tk.TOP, padx=10)

        # # Output File Entry
        # self.output_file_entry_crawler = tk.Entry(
        #     self.news_crawler_frame, bg=self.entry_bg, fg=self.entry_fg)
        # self.output_file_entry_crawler.pack(side=tk.TOP, padx=10)

        # Button
        self.run_news_crawler_button = tk.Button(
            self.news_crawler_frame, text="Run News Crawler", command=self.run_news_crawler,
            bg=self.button_bg, fg=self.button_fg)
        self.run_news_crawler_button.pack(side=tk.TOP, padx=10)

        # Sentiment Analysis

        # Frame
        self.sentiment_analysis_frame = tk.Frame(
            self.button_frame, bg="#333333")
        self.sentiment_analysis_frame.pack(side=tk.LEFT, pady=10)

        # # Input File Label
        # self.input_file_label_sentiment = tk.Label(
        #     self.sentiment_analysis_frame, text="Input JSON File (Sentiment Analysis):", fg=self.label_fg, bg="#333333")
        # self.input_file_label_sentiment.pack(side=tk.TOP, padx=10)

        # # Input File Entry
        # self.input_file_entry_sentiment = tk.Entry(
        #     self.sentiment_analysis_frame, bg=self.entry_bg, fg=self.entry_fg)
        # self.input_file_entry_sentiment.pack(side=tk.TOP, padx=10)

        # # Output File Label
        # self.output_file_label_sentiment = tk.Label(
        #     self.sentiment_analysis_frame, text="Output JSON File (Sentiment Analysis):", fg=self.label_fg, bg="#333333")
        # self.output_file_label_sentiment.pack(side=tk.TOP, padx=10)

        # # Output File Entry
        # self.output_file_entry_sentiment = tk.Entry(
        #     self.sentiment_analysis_frame, bg=self.entry_bg, fg=self.entry_fg)
        # self.output_file_entry_sentiment.pack(side=tk.TOP, padx=10)

        # Button
        self.run_sentiment_analysis_button = tk.Button(
            self.sentiment_analysis_frame, text="Run Sentiment Analysis", command=self.run_sentiment_analysis,
            bg=self.button_bg, fg=self.button_fg)
        self.run_sentiment_analysis_button.pack(side=tk.TOP, padx=10)

        # Sentiment Processing

        # Frame
        self.sentiment_processing_frame = tk.Frame(
            self.button_frame, bg="#333333")
        self.sentiment_processing_frame.pack(side=tk.LEFT, pady=10)

        # # Input File Label
        # self.input_file_label_processing = tk.Label(
        #     self.sentiment_processing_frame, text="Input JSON File (Sentiment Processing):", fg=self.label_fg, bg="#333333")
        # self.input_file_label_processing.pack(side=tk.TOP, padx=10)

        # # Input File Entry
        # self.input_file_entry_processing = tk.Entry(
        #     self.sentiment_processing_frame, bg=self.entry_bg, fg=self.entry_fg)
        # self.input_file_entry_processing.pack(side=tk.TOP, padx=10)

        # # Output File Label
        # self.output_file_label_processing = tk.Label(
        #     self.sentiment_processing_frame, text="Output JSON File (Sentiment Processing):", fg=self.label_fg, bg="#333333")
        # self.output_file_label_processing.pack(side=tk.TOP, padx=10)

        # # Output File Entry
        # self.output_file_entry_processing = tk.Entry(
        #     self.sentiment_processing_frame, bg=self.entry_bg, fg=self.entry_fg)
        # self.output_file_entry_processing.pack(side=tk.TOP, padx=10)

        # Button
        self.run_sentiment_processing_button = tk.Button(
            self.sentiment_processing_frame, text="Run Sentiment Processing", command=self.run_sentiment_processing,
            bg=self.button_bg, fg=self.button_fg)
        self.run_sentiment_processing_button.pack(side=tk.TOP, padx=10)

        # Terminal
        self.terminal_text = tk.Text(
            self.root, height=10, width=50, bg=self.entry_bg, fg=self.text_color)
        # Use pack with fill and expand options
        self.terminal_text.pack(fill=tk.BOTH, expand=True)

        # Exit
        self.exit_button = tk.Button(
            self.root, text="Exit", command=self.root.quit, bg=self.button_bg, fg=self.button_fg)
        self.exit_button.pack(pady=10)

    def load_config(self):
        """
        Loads configuration from the 'config.ini' file.
        """
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

    def get_script_path(self, script_name):
        """
        Retrieves the script path from the configuration.

        Args:
            script_name (str): Name of the script.

        Returns:
            str: Path to the script.
        """
        return self.config["Scripts"][script_name]

    def run_script(self, script_name):
        """
        Runs a script and displays output in the terminal.

        Args:
            script_name (str): Name of the script to run.
            output_entry (Entry): The entry widget containing the output file path.
        """
        try:
            script_path = self.get_script_path(script_name)

            self.terminal_text.delete(1.0, tk.END)
            self.add_to_terminal(f"Running {script_name}...\n")

            def run_script_in_thread():
                process = subprocess.Popen(
                    ["python", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line-buffered output
                    universal_newlines=True
                )

                # Read and update terminal in real-time
                while True:
                    output_line = process.stdout.readline()
                    if not output_line:
                        break
                    self.add_to_terminal(output_line)
                    self.root.update_idletasks()  # Update the display

                process.stdout.close()
                process.wait()  # Wait for the process to finish

            # Run the script in a separate thread
            script_thread = threading.Thread(target=run_script_in_thread)
            script_thread.start()

        except Exception as e:
            self.add_to_terminal(f"Error: {str(e)}")

    def browse_file(self, entry):
        """
        Opens a file dialog to select a JSON file and updates the entry.

        Args:
            entry (Entry): The entry widget to update.
        """
        file = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json")])
        entry.delete(0, tk.END)
        entry.insert(0, file)

    def run_news_crawler(self):
        """
        Runs the news crawler script.
        """
        self.run_script("news_crawler")

    def run_sentiment_analysis(self):
        """
        Runs the sentiment analysis script.
        """
        self.run_script("sentiment_analysis")

    def run_sentiment_processing(self):
        """
        Runs the sentiment processing script.
        """
        self.run_script("sentiment_processing")

    def add_to_terminal(self, text):
        """
        Adds text to the terminal Text widget.

        Args:
            text (str): The text to add.
        """
        self.terminal_text.insert(tk.END, text)
        self.terminal_text.see(tk.END)
        self.root.update_idletasks()  # Update the display


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
