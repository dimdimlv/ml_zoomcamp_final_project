from typing import Tuple, Optional
from types import TracebackType
from src.logger import logging

def error_message_detail(error, error_detail: Tuple[Optional[type], Optional[Exception], Optional[TracebackType]]):
    _, _, exc_tb = error_detail
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "unknown"
        line_number = 0
    error_message = f"Error occurred in script: {file_name} at line number: {line_number} with message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: Tuple[Optional[type], Optional[Exception], Optional[TracebackType]]):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    