from time import time, ctime


def output_time():
    t = time()
    return str(ctime(t))