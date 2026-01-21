from pathlib import Path
from dfastbe.io.logger import LogData
from dfastbe import __path__

log_data = LogData(Path(__path__[0]) / "io/log_data/messages.UK.ini")