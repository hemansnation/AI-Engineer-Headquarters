import os
import sys

class InsuranceException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message = InsuranceException.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{error_message}]"

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return str(InsuranceException)


class InsuranceClaimException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message = InsuranceClaimException.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{error_message}]"

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return str(InsuranceClaimException)
