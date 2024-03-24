class Error(Exception):
    pass


class UserNotFoundException(Exception):
    def __init__(self, user: str, message: str, cause: Exception = None) -> None:
        super().__init__(message)
        self.user = user
        self.cause = cause
        self.message = message

    def __str__(self) -> str:
        message = f"Error with user {self.user}\n{self.message}"
        if self.cause:
            message += f"\nCaused by: {self.cause}"
        return message


class PredictionException(Exception):
    def __init__(self, message: str, cause: Exception = None) -> None:
        super().__init__(message)
        self.cause = cause
        self.message = message

    def __str__(self) -> str:
        message = f"{self.message}"
        if self.cause:
            message += f"\nCaused by: {self.cause}"
        return message
