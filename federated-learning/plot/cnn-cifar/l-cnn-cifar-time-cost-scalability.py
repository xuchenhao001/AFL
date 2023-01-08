import sys

from plot.utils.time_acc_base import plot_time_acc_nodes

x_range = [10, 20, 50, 100]

dbafl = [18.782, 19.0468, 19.3684, 19.9554]
bsfl = [56.6484, 79.8439, 139.3137, 240.73906]
aso_fed = [17.3484, 17.6132, 17.9348, 18.5218]
bdfl = [33.574, 33.8388, 34.1604, 34.7474]
apfl = [100.6498, 100.9146, 101.2362, 101.8232]
fed_avg = [59.4668, 82.6623, 142.1321, 243.55746]
local_train = [14.1838, 14.1838, 14.1838, 14.1838]

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_time_acc_nodes("", x_range, dbafl, bsfl, aso_fed, bdfl, apfl, fed_avg, local_train, save_path, plot_size="L")
