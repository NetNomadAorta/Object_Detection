import time
import sys
for x in range (0,5):  
    b = "Loading" + "." * x
    sys.stdout.write('\033[2K\033[1G')
    print (b, end="\r")
    time.sleep(1)