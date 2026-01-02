from typing import Tuple, Optional, Any
from types import TracebackType
from src.logger import logging

def error_message_detail(error: Exception, error_detail: Any):
    """Builds a rich error message even if error_detail is not exc_info.

    Accepts either sys.exc_info() tuple or a module with exc_info (e.g., sys).
    """
    try:
        if hasattr(error_detail, "exc_info"):
            error_detail = error_detail.exc_info()
        if isinstance(error_detail, tuple):
            _, _, exc_tb = error_detail
        else:
            exc_tb = None
    except Exception:
        exc_tb = None

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "unknown"
        line_number = 0

    error_message = (
        f"Error occurred in script: {file_name} at line number: {line_number} "
        f"with message: {str(error)}"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: Any):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    