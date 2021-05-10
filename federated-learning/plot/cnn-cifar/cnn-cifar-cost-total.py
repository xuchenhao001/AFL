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
fed_server = [9.23, 9.94, 8.9, 8.81, 8.49, 12.37, 9.07, 9.12, 6.96, 9.26, 10.15, 9.9, 8.4, 9.06, 9.13, 9.68, 9.21, 6.9, 8.86, 9.91, 8.61, 7.7, 9.78, 9.25, 7.66, 10.73, 7.16, 7.2, 8.4, 9.56, 7.65, 9.41, 9.11, 9.41, 11.37, 10.26, 9.54, 8.85, 10.51, 9.54, 10.39, 8.04, 7.3, 9.2, 8.04, 7.89, 9.79, 8.95, 9.7, 9.12]
main_fed_localA = [1.7, 22.38, 2.01, 1.88, 1.96, 2.02, 1.9, 1.91, 1.8, 1.84, 1.8, 29.51, 1.76, 1.84, 1.96, 1.93, 1.88, 1.8, 1.8, 1.84, 1.84, 29.56, 1.84, 1.77, 1.76, 1.88, 1.91, 1.79, 1.73, 1.82, 1.78, 30.91, 2.05, 1.86, 1.92, 1.94, 1.84, 1.85, 1.94, 2.05, 1.89, 30.56, 1.92, 1.93, 1.89, 1.8, 1.82, 1.82, 1.82, 1.83]
main_fed = [7.07, 3.95, 4.01, 4.2, 4.15, 4.09, 4.11, 4.1, 4.19, 4.08, 4.04, 4.17, 4.05, 4.29, 4.18, 3.96, 4.07, 4.19, 4.11, 4.14, 4.02, 3.98, 4.11, 4.07, 3.86, 3.92, 4.35, 4.09, 4.16, 4.1, 4.2, 3.98, 4.01, 4.05, 4.01, 4.05, 4.23, 4.05, 4.13, 4.06, 4.2, 4.24, 4.11, 4.2, 4.07, 3.92, 3.91, 4.0, 4.16, 2.95]
main_nn = [0.83, 0.83, 0.89, 0.89, 0.89, 0.89, 0.93, 0.87, 0.89, 0.92, 0.91, 0.92, 0.93, 0.91, 0.94, 0.93, 0.88, 0.85, 0.84, 0.91, 0.97, 0.92, 0.9, 0.94, 0.96, 0.9, 0.89, 0.9, 0.9, 0.91, 0.92, 0.97, 0.96, 0.94, 0.89, 0.88, 0.88, 0.95, 0.96, 0.93, 0.87, 0.88, 0.93, 0.87, 0.92, 0.83, 0.83, 0.8, 0.82, 0.85]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, fed_server, label="SCEI with negotiated α", linewidth=3)
axes.plot(x, main_nn, label="Local Training", linestyle='--', alpha=0.5)
axes.plot(x, main_fed_localA, label="APFL", linestyle='--', alpha=0.5)
axes.plot(x, main_fed, label="FedAvg", linestyle='--', alpha=0.5)


axes.set_xlabel("Training Rounds", **csXYLabelFont)
axes.set_ylabel("Total Time Consumption (s)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
# plt.ylim(90, 100)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
