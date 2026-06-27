import logging
import sys
from app.config import settings

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.
    
    Args:
        name (str): The name of the module (typically __name__)
        
    Returns:
        logging.Logger: The configured logger object
    """
    logger = logging.getLogger(name)
    
    # Convert string log level from settings to logging constant (default to INFO)
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Ensure we don't add multiple handlers if get_logger is called multiple times for the same name
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Format: timestamp | level | module | message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger
