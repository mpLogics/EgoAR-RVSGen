from datetime import datetime


class StoreLogs():
    def __init__(self):
        self.msg = "Error Detected"
        self.des = "Logging error"
        self.fileName = "Logs.txt"
        self.mode = "a"

    def LogMessages(self):    
        now = datetime.now()
        f = open(self.fileName, self.mode)
        f.write(self.msg + "\nTime : " + (str)(now) + "\nDescription: " + self.des +"\n\n")
        f.close()
    