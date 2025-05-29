import sys
import traceback

class InsuranceClaimException(Exception):
    """
    Custom exception class for the Insurance Claim Prediction project.
    Captures detailed information about exceptions, including the error message,
    file name, and line number where the exception occurred.
    """

    def __init__(self, error_message: str, error_detail: sys):
        """
        Initializes the InsuranceClaimException instance with detailed error information.

        Args:
            error_message (str): The original error message.
            error_detail (sys): The sys module, used to extract exception details.
        """
        super().__init__(error_message)
        self.error_message = error_message

        # Extract exception information
        _, _, exc_tb = error_detail.exc_info()


        if exc_tb is not None:
            # Traverse to the innermost traceback to get accurate error location
            while exc_tb.tb_next:
                exc_tb = exc_tb.tb_next

            self.filename = exc_tb.tb_frame.f_code.co_filename  # File where exception occurred
            self.lineno = exc_tb.tb_lineno  # Line number where exception occurred

    def __str__(self):
        """
        Returns a formatted string representation of the error, including
        the error message, file name, and line number.

        Returns:
            str: Formatted error message.
        """
        return f"Error: {self.error_message} | File: {self.filename} | Line: {self.lineno}"


if __name__ == "__main__":
    try:
        # Example operation that will raise a ZeroDivisionError
        result = 1 / 0
    except Exception as e:
        # Raise custom exception with detailed information
        raise InsuranceClaimException(str(e), sys)
