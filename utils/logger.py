import logging
import sys


class LoggerSetup:
    def __init__(
        self,
        name="hand_eye_calibration",
        log_file="calibration_run.log",
        file_level=logging.DEBUG,
        console_level=logging.INFO,
    ):
        """
        Initializes the LoggerSetup.

        Args:
            name (str): The name of the logger. Using a common name allows different
                        modules to share the same logger instance.
            log_file (str): The path to the log file.
            file_level (int): The logging level for the file handler (e.g., logging.DEBUG).
            console_level (int): The logging level for the console handler (e.g., logging.INFO).
        """
        self.name = name
        self.log_file = log_file
        self.file_level = file_level
        self.console_level = console_level
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)

    def setup_logger(self):
        """
        Configures and returns the logger instance.

        It adds handlers only if the logger doesn't already have them,
        preventing duplicate log messages.

        Returns:
            logging.Logger: The configured logger instance.
        """
        # Prevent adding handlers multiple times to the same logger instance
        if not self.logger.handlers:
            # --- File Handler ---
            # This handler writes all messages from DEBUG level and up to a file.
            file_handler = logging.FileHandler(
                self.log_file, mode="w"
            )  # 'w' to overwrite the log file on each run
            file_handler.setLevel(self.file_level)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # --- Console Handler ---
            # This handler prints messages from INFO level and up to the console.
            # console_handler = logging.StreamHandler(sys.stdout)
            # console_handler.setLevel(self.console_level)
            # console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            # console_handler.setFormatter(console_formatter)
            # self.logger.addHandler(console_handler)

        return self.logger
