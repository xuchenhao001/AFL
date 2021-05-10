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
fed_server = [96.87, 97.8, 98.33, 98.27, 98.67, 98.87, 98.6, 98.93, 99.13, 98.87, 98.87, 98.8, 99.07, 99.07, 98.93, 99.07, 99.07, 99.2, 99.07, 99.27, 99.2, 99.07, 99.0, 99.07, 99.13, 99.13, 99.0, 99.13, 99.27, 99.07, 99.2, 99.07, 99.33, 99.47, 99.2, 99.47, 99.2, 99.4, 99.47, 99.53, 99.4, 99.2, 99.33, 99.53, 99.47, 99.47, 99.6, 99.27, 99.47, 99.27]
main_fed_localA = [6.47, 87.27, 98.07, 98.4, 98.8, 98.8, 98.87, 99.13, 99.13, 98.93, 99.07, 99.0, 99.0, 99.0, 98.93, 99.07, 99.0, 98.8, 99.13, 99.0, 98.93, 98.93, 99.13, 99.2, 99.2, 99.27, 99.2, 99.0, 99.13, 99.13, 99.2, 99.2, 99.07, 99.13, 99.0, 99.07, 99.07, 98.93, 99.13, 99.13, 99.07, 99.07, 98.93, 98.93, 98.93, 99.07, 99.13, 99.07, 99.13, 99.2]
main_fed = [62.33, 68.0, 73.8, 78.47, 81.2, 83.33, 84.47, 85.07, 85.33, 85.6, 86.47, 86.8, 88.27, 88.33, 88.47, 88.8, 89.6, 89.67, 89.67, 89.67, 90.27, 90.87, 90.87, 91.07, 91.13, 90.8, 91.6, 91.6, 91.8, 91.67, 91.93, 92.0, 92.0, 92.07, 92.13, 92.13, 92.8, 92.53, 92.8, 92.87, 92.73, 92.53, 92.93, 93.2, 93.4, 94.0, 93.67, 93.8, 93.67, 94.2]
main_nn = [6.47, 97.07, 98.07, 98.27, 98.73, 99.07, 98.6, 98.93, 99.07, 99.2, 98.93, 98.93, 99.13, 99.07, 99.13, 99.0, 99.07, 99.13, 99.0, 99.13, 99.0, 99.13, 99.13, 98.93, 99.0, 99.0, 99.13, 99.0, 98.93, 99.27, 98.93, 98.93, 99.0, 99.13, 99.0, 99.0, 99.07, 99.13, 99.0, 99.0, 99.13, 99.2, 99.13, 99.0, 99.0, 99.0, 99.2, 99.13, 99.0, 99.13]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, fed_server, label="SCEI with negotiated α", linewidth=3)
axes.plot(x, main_nn, label="Local Training", linestyle='--', alpha=0.5)
axes.plot(x, main_fed_localA, label="APFL", linestyle='--', alpha=0.5)
axes.plot(x, main_fed, label="FedAvg", linestyle='--', alpha=0.5)
# axes.plot(x, scei, label="SCEI with negotiated α", linewidth=3, color='#1f77b4')


axes.set_xlabel("Training Rounds", **csXYLabelFont)
axes.set_ylabel("Mean of Local Test Accuracy (%)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
plt.ylim(90, 100)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
