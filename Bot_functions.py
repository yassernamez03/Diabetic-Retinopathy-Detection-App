from functions import *
from datetime import datetime

    
def TimeQuery():
    time = datetime.now().strftime("%H:%M:%S")
    return " It's {}".format(time)

def DateQuery():
    time = datetime.now().strftime("%Y-%m-%d")
    return " Today is {}".format(time)

