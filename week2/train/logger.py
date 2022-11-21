import logging

from rich.logging import RichHandler
from rich.traceback import install


class Logger:
    def __init__(self, name, log_level: int = logging.DEBUG) -> None:
        install(show_locals=True)
        logging.basicConfig(
            level=log_level,
            format="%(module)s:%(name)s: %(lineno)s - %(message)s",
            handlers=[RichHandler()],
        )
        logger = logging.getLogger(name)

        self.log = logger

