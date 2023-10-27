import logging
from  datetime import datetime
def configure_logger(path):
    """ Configures logger used throughout the application
        To get an instance of the configured logger from any module, simply call:
            `logger = logging.getLogger(__name__)`
    """

    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(filename = path + f"out_{dt_string}.log"),
            logging.StreamHandler()
        ]
    )
