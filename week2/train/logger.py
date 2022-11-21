import logging

from rich.logging import RichHandler



logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(filename)s:%(lineno)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("docker_tutorial")



