import time


class Engine:
    def __init__(self):
        self.container = ""

    def mock_request(self):
        time.sleep(1)
        status = "warm"
        if self.container == "":
            self.container = "mock"
            time.sleep(2)
            status = "cold"

        return status
