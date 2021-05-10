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
fed_server = [7.92, 8.52, 7.7, 7.09, 7.12, 9.02, 7.87, 7.84, 5.73, 8.0, 8.56, 8.54, 6.73, 7.81, 7.92, 8.2, 7.77, 5.62, 7.0, 8.48, 7.05, 6.44, 8.46, 8.04, 6.41, 9.06, 5.91, 5.92, 6.99, 8.26, 6.37, 8.16, 7.09, 8.07, 8.76, 8.69, 8.1, 6.97, 8.87, 8.26, 8.93, 6.71, 6.03, 7.32, 6.71, 6.63, 8.44, 7.34, 8.45, 6.95]
main_fed_localA = [0.08, 20.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 27.39, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 27.63, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.51, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
main_fed = [6.24, 3.01, 3.13, 3.26, 3.19, 3.13, 3.21, 3.19, 3.29, 3.17, 3.15, 3.26, 3.15, 3.33, 3.28, 3.08, 3.19, 3.27, 3.2, 3.22, 3.14, 3.11, 3.17, 3.17, 2.99, 3.04, 3.44, 3.23, 3.26, 3.19, 3.3, 3.09, 3.12, 3.14, 3.11, 3.17, 3.33, 3.16, 3.23, 3.17, 3.28, 3.32, 3.22, 3.31, 3.16, 3.04, 3.03, 3.13, 3.25, 2.06]

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
