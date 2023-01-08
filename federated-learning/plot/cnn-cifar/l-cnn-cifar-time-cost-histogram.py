import sys

from plot.utils.time_acc_base import plot_time_historgram

# iot-2-network
fed_async = [26.52, 0.25, 3.07, 0, 29.82]
fed_avg = [23.96, 0.23, 0.05, 34.55, 58.79]
fed_localA = [48.84, 0.21, 0.01, 54.1, 103.16]
fed_sync = [23.91, 0.24, 2.52, 30.52, 57.19]
local_train = [24.15, 0.2, 0.0, 0.01, 24.36]
fed_asofed = [27.62, 0.23, 0.06, 0.94, 28.85]
fed_bdfl = [28.99, 0.24, 4.55, 0.0, 33.08]

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_time_historgram("", fed_async, fed_avg, fed_sync, fed_localA, local_train, fed_asofed, fed_bdfl, save_path, plot_size="L")
