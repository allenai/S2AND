import logging
from importlib.metadata import version as package_version

logger = logging.getLogger("s2and")
logger.addHandler(logging.NullHandler())

__version__ = package_version("s2and")
