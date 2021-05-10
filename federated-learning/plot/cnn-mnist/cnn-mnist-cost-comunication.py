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

fed_server = [6.55, 8.08, 8.22, 8.28, 6.59, 6.55, 8.05, 6.25, 8.51, 8.53, 7.87, 6.58, 8.32, 8.14, 8.37, 8.57, 6.61, 7.21, 6.76, 7.59, 6.76, 6.9, 8.26, 7.22, 7.8, 7.61, 7.74, 7.04, 7.48, 8.32, 6.25, 6.83, 8.85, 8.42, 6.59, 6.86, 8.5, 7.57, 7.28, 7.75, 7.24, 6.89, 7.14, 7.18, 7.13, 8.81, 6.95, 6.85, 6.76, 7.05]
main_fed_localA = [0.06, 19.28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.84, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.84, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
main_fed = [5.14, 1.84, 1.88, 1.9, 1.78, 1.94, 1.81, 1.82, 1.85, 1.9, 1.86, 1.89, 1.92, 2.09, 2.12, 1.79, 1.82, 1.76, 1.88, 1.9, 1.92, 1.93, 2.04, 1.82, 1.9, 1.82, 1.96, 1.78, 1.94, 1.93, 1.77, 1.95, 1.78, 2.08, 1.98, 1.83, 1.8, 1.89, 1.95, 1.87, 1.94, 1.88, 1.98, 1.88, 1.92, 1.86, 1.9, 1.94, 1.98, 0.99]

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
