import os
import datetime

import numpy as np

class Logger:
    def __init__(
        self,
        filename: str,
        filepath: str = "log"
    ):
       
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.filepath = filepath + "/"

        self.filename = filename + ".txt"

    def log(self, message):
        try:
            with open(self.filepath + self.filename, 'a') as file:
                if isinstance(message, dict):
                    for key, value in message.items():
                        file.write(f"{str(key)}: {str(value)}\n")
                elif isinstance(message, list) or isinstance(message, np.ndarray):
                    file.write("[ ")
                    for value in message:
                        file.write(str(value))
                        file.write(" ")
                    file.write("]")
                else:
                    file.write(message)
                
                file.write('\n')
        except Exception as e:
            self.error_message(e, message)

    def error_message(self, e, message):
        print("Error: Logger.write")

        print("-" * 50 + "error message" + "-" * 50)
        print(e)

        print("-" * 50 + "failed to write this message" + "-" * 50)
        print(message)
        print("Type: ", type(message))

        print("Error end", end="\n\n")

    def timestamp(self):
        with open(self.filepath + self.filename, 'a') as file:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file.write(timestamp + '\n')