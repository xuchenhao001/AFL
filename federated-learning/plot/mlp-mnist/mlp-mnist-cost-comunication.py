# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
fed_server = [7.26, 7.0, 7.2, 7.91, 8.07, 8.73, 7.84, 8.47, 7.6, 8.16, 7.83, 7.41, 8.16, 10.29, 8.05, 7.87, 7.79, 7.88, 7.41, 7.77, 8.1, 7.94, 7.62, 8.1, 7.63, 7.71, 8.33, 8.04, 10.99, 8.81, 7.95, 7.6, 7.54, 8.16, 7.58, 8.29, 7.95, 7.73, 8.65, 7.69, 8.11, 7.75, 7.59, 7.97, 7.79, 7.79, 7.5, 7.89, 7.65, 7.41]
main_fed_localA = [0.06, 20.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.72, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.63, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
main_fed = [5.42, 2.34, 2.38, 2.36, 2.25, 2.33, 2.18, 2.22, 2.23, 2.22, 2.31, 2.34, 2.33, 2.24, 2.31, 2.18, 2.21, 2.37, 2.2, 2.24, 2.34, 2.24, 2.23, 2.33, 2.23, 2.27, 2.37, 2.3, 2.26, 2.37, 2.27, 2.33, 2.33, 2.26, 2.2, 2.3, 2.22, 2.33, 2.29, 2.19, 2.19, 2.29, 2.22, 2.29, 2.23, 2.21, 2.2, 2.23, 2.19, 1.53]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, fed_server, label="SCEI with negotiated α", linewidth=3)
axes.plot(x, main_fed, label="FedAvg", linestyle='--', alpha=0.5)
axes.plot(x, main_fed_localA, label="APFL", linestyle='--', alpha=0.5)


axes.set_xlabel("Training Rounds", **csXYLabelFont)
axes.set_ylabel("Communication Time Consumption (s)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
# plt.ylim(90, 100)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
