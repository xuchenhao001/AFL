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

fed_server = [7.92, 7.66, 7.86, 8.64, 8.93, 9.8, 8.7, 9.42, 8.35, 8.99, 8.58, 8.2, 8.85, 11.73, 8.82, 8.63, 8.7, 8.61, 8.28, 8.66, 9.34, 8.61, 8.39, 9.17, 8.35, 8.84, 9.47, 9.1, 12.13, 10.13, 8.74, 8.45, 8.33, 9.14, 8.29, 9.34, 8.7, 8.47, 9.81, 8.46, 8.79, 8.43, 8.24, 8.71, 8.48, 8.47, 8.19, 8.57, 8.36, 8.11]
main_fed_localA = [0.84, 21.1, 0.97, 1.01, 0.95, 0.88, 0.84, 0.88, 0.82, 0.88, 0.83, 24.0, 1.01, 0.87, 0.84, 0.86, 0.88, 0.84, 0.83, 0.92, 0.82, 23.61, 0.86, 0.9, 0.83, 0.84, 0.86, 0.86, 0.87, 0.84, 0.8, 20.52, 0.85, 0.84, 0.9, 0.9, 0.83, 0.84, 0.91, 0.87, 0.86, 20.24, 0.86, 0.9, 0.86, 0.85, 0.89, 0.81, 0.87, 0.81]
main_fed = [5.83, 2.79, 2.83, 2.8, 2.69, 2.77, 2.61, 2.67, 2.66, 2.65, 2.76, 2.79, 2.76, 2.67, 2.76, 2.62, 2.65, 2.81, 2.66, 2.67, 2.79, 2.67, 2.67, 2.77, 2.67, 2.72, 2.81, 2.76, 2.68, 2.81, 2.72, 2.78, 2.78, 2.69, 2.63, 2.74, 2.66, 2.77, 2.73, 2.65, 2.63, 2.74, 2.65, 2.73, 2.68, 2.64, 2.64, 2.67, 2.63, 1.97]
main_nn = [0.45, 0.43, 0.41, 0.41, 0.41, 0.43, 0.44, 0.43, 0.41, 0.43, 0.44, 0.47, 0.41, 0.41, 0.43, 0.48, 0.44, 0.42, 0.42, 0.43, 0.43, 0.44, 0.42, 0.47, 0.44, 0.44, 0.45, 0.41, 0.44, 0.43, 0.45, 0.43, 0.43, 0.44, 0.43, 0.42, 0.45, 0.42, 0.42, 0.43, 0.46, 0.41, 0.44, 0.44, 0.42, 0.43, 0.41, 0.41, 0.43, 0.41]

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
