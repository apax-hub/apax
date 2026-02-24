import csv
import threading
import queue
import time
import os

class ThreadedCSVLogger:
    def __init__(self, filepath, fieldnames=None, flush_interval=1.0):
        """
        Initialize the ThreadedCSVLogger.
        
        :param filepath: Path to the CSV file where logs will be saved.
        :param fieldnames: List of fieldnames (keys) for the CSV file. If not provided,
                           fieldnames will be inferred from the first logged metrics.
        :param flush_interval: Time interval (in seconds) to flush the log to the file.
        """
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.flush_interval = flush_interval
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._writer_thread)
        self.csv_file = None
        self.writer = None
        self.headers_written = False
        self.last_epoch = -1  # Track the last logged epoch

        # Check if the file already exists and recover the last epoch
        if os.path.exists(self.filepath):
            self._recover_last_epoch()

        # Start the background thread for logging
        self.thread.start()

    def _recover_last_epoch(self):
        """Recover the last logged epoch from the CSV file if it exists."""
        try:
            with open(self.filepath, 'r') as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row and 'epoch' in last_row:
                    self.last_epoch = int(last_row['epoch'])
                self.fieldnames = reader.fieldnames
                self.headers_written = True  # Headers were already written
        except (csv.Error, FileNotFoundError):
            pass  # Handle potential CSV read errors if the file is corrupted or does not exist

    def log(self, metrics_dict):
        """
        Log a dictionary of metrics to the CSV file asynchronously.
        
        :param metrics_dict: Dictionary containing metric names as keys and their values.
        """
        self.queue.put(metrics_dict)

    def _writer_thread(self):
        """
        Internal method that runs in a separate thread, writing metrics to the CSV file.
        """
        # Open the file in append mode
        with open(self.filepath, mode='a', newline='') as csv_file:
            self.csv_file = csv_file
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)

            # Write the headers if needed
            if not self.headers_written and self.fieldnames:
                self.writer.writeheader()
                self.headers_written = True

            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    # Try to get a batch of logs from the queue (wait max flush_interval)
                    metrics = self.queue.get(timeout=self.flush_interval)
                    self._write_row(metrics)
                except queue.Empty:
                    pass  # Timeout reached, no new logs, but continue running

                # Optionally flush the file
                self.csv_file.flush()

    def _write_row(self, metrics_dict):
        """
        Write a single row of metrics to the CSV file.
        
        :param metrics_dict: Dictionary containing metric names and values.
        """
        if not self.fieldnames:
            self.fieldnames = list(metrics_dict.keys())
            self.writer.fieldnames = self.fieldnames

            # Write headers once we have the fieldnames
            if not self.headers_written:
                self.writer.writeheader()
                self.headers_written = True
        
        self.writer.writerow(metrics_dict)

    def close(self):
        """
        Signal the logger to stop and wait for the background thread to finish.
        """
        self.stop_event.set()
        self.thread.join()

# Example Usage:
if __name__ == "__main__":
    # Create a logger that will append to the file or resume from where it left off
    logger = ThreadedCSVLogger('training_metrics_threaded.csv')

    # Retrieve the last logged epoch to continue from there
    start_epoch = logger.last_epoch + 1
    print(f"Resuming training from epoch {start_epoch}...")

    # Simulate logging metrics asynchronously during training
    for epoch in range(start_epoch, start_epoch + 5):
        metrics = {
            'epoch': epoch,
            'loss': 0.5 - epoch * 0.1,
            'accuracy': 0.7 + epoch * 0.05
        }
        logger.log(metrics)
        time.sleep(0.5)  # Simulate time between training steps

    # Close the logger, ensuring all data is written
    logger.close()