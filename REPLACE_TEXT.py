import time
import sys


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )



# Starting stopwatch to see how long process takes
start_time = time.time()

for x in range (0,5):  
    b = "Loading" + "." * x
    sys.stdout.write('\033[2K\033[1G')
    print (b, end="\r")
    time.sleep(1)
# To make a new line so next print starts its own line
print("")



print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)
