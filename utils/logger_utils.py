import os
import sys
import logging

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

def setup_logging(log_path: str, log_filename: str = "log.txt"):
    os.makedirs(log_path, exist_ok=True)
    log_file_path = os.path.join(log_path, log_filename)

    # Open log file
    log_file = open(log_file_path, "w")

    # Redirect stdout and stderr to both terminal and file
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    # Setup logging module to also log to both
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"Logging initialized. Output will go to terminal and {log_file_path}")
