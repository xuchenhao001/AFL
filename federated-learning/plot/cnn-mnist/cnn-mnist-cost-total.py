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
fed_server = [7.8, 9.28, 9.41, 9.51, 7.96, 7.95, 9.31, 7.38, 9.81, 9.99, 8.94, 7.78, 9.7, 9.36, 9.61, 9.84, 7.86, 8.52, 8.0, 9.17, 8.31, 8.32, 9.51, 8.45, 10.48, 9.69, 10.46, 8.41, 9.05, 10.23, 7.81, 9.28, 10.14, 9.71, 7.68, 8.25, 9.83, 8.85, 8.62, 9.46, 8.68, 8.25, 8.54, 8.45, 8.35, 10.03, 8.26, 8.07, 7.82, 8.69]
main_fed_localA = [1.48, 21.01, 1.65, 1.64, 1.66, 1.63, 1.61, 1.6, 1.66, 1.59, 1.52, 19.44, 1.62, 1.55, 1.52, 1.5, 1.53, 1.55, 1.56, 1.5, 1.49, 18.45, 1.54, 1.53, 1.5, 1.55, 1.54, 1.53, 1.56, 1.5, 1.43, 18.78, 1.55, 1.49, 1.47, 1.48, 1.52, 1.51, 1.49, 1.49, 1.49, 18.45, 1.55, 1.5, 1.52, 1.57, 1.53, 1.51, 1.5, 1.49]
main_fed = [5.89, 2.62, 2.69, 2.72, 2.59, 2.75, 2.61, 2.66, 2.66, 2.7, 2.65, 2.72, 2.76, 2.91, 2.9, 2.61, 2.61, 2.57, 2.7, 2.71, 2.73, 2.73, 2.85, 2.61, 2.7, 2.59, 2.74, 2.58, 2.75, 2.74, 2.56, 2.74, 2.61, 2.87, 2.79, 2.63, 2.6, 2.66, 2.78, 2.69, 2.73, 2.68, 2.75, 2.69, 2.71, 2.67, 2.71, 2.73, 2.79, 1.8]
main_nn = [0.76, 0.76, 0.79, 0.77, 0.78, 0.78, 0.8, 0.78, 0.81, 0.79, 0.84, 0.77, 0.78, 0.8, 0.75, 0.78, 0.78, 0.78, 0.78, 0.76, 0.76, 0.77, 0.79, 0.79, 0.76, 0.78, 0.79, 0.79, 0.81, 0.82, 0.77, 0.76, 0.83, 0.77, 0.82, 0.82, 0.8, 0.77, 0.78, 0.79, 0.79, 0.8, 0.8, 0.81, 0.78, 0.79, 0.78, 0.8, 0.74, 0.74]

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
