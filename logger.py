import logging
from config import log_level

# Specifying log format for convenience
logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s", 
    datefmt="%d/%m/%Y %H:%M:%S", 
    level=log_level
)
logger = logging.getLogger()
