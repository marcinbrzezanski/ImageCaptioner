import logging
from colorama import Fore, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)

class Logger:
    """Logger class with colorized output."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def __init__(self, level: int = logging.DEBUG):
        """
        Initialize the logger.
        :param level: Logging level (default: DEBUG).
        """
        self.level = level
        self.logger = None

    def _get_logger(self):
        """Initialize the logger only when needed."""
        if self.logger is None:
            self.logger = logging.getLogger("CustomLogger")
            self.logger.setLevel(self.level)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)

            # Set a formatter with colors
            formatter = self.ColoredFormatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
            console_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)

        return self.logger

    class ColoredFormatter(logging.Formatter):
        """Custom formatter to add colors to log messages."""
        def format(self, record):
            level_color = Logger.LEVEL_COLORS.get(record.levelno, "")
            record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"
            record.msg = f"{level_color}{record.msg}{Style.RESET_ALL}"
            return super().format(record)

    def debug(self, message: str):
        self._get_logger().debug(message)

    def info(self, message: str):
        self._get_logger().info(message)

    def warning(self, message: str):
        self._get_logger().warning(message)

    def error(self, message: str):
        self._get_logger().error(message)

    def critical(self, message: str):
        self._get_logger().critical(message)

# Usage example:
logger = Logger()