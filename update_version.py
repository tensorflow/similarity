import shutil
import time

with open("setup.py", "rt") as i:
    with open("setup.py_new", "wt") as o:
        for line in i.readlines():
            tokens = line.split(" ")
            if tokens[0] == 'MINOR_VERSION':
                minor_version = int(tokens[2]) + 1
                line = "MINOR_VERSION = %d\n" % minor_version
            o.write(line)

shutil.move("setup.py_new", "setup.py")
