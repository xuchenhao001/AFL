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

fed_server = [95.87, 96.8, 97.07, 97.33, 97.4, 97.4, 97.33, 97.07, 97.47, 97.2, 97.2, 97.2, 97.33, 97.33, 97.4, 97.4, 97.4, 97.27, 97.27, 97.33, 97.4, 97.33, 97.4, 97.4, 97.47, 97.27, 97.4, 97.4, 97.33, 97.47, 97.53, 97.4, 97.67, 97.47, 97.4, 97.67, 97.47, 97.6, 97.67, 97.53, 97.53, 97.47, 97.53, 97.47, 97.67, 97.53, 97.67, 97.47, 97.6, 97.73]
main_fed_localA = [9.93, 89.73, 96.6, 97.07, 97.27, 97.47, 97.4, 97.67, 97.47, 97.67, 97.53, 95.2, 97.8, 97.53, 97.73, 97.67, 97.53, 97.53, 97.73, 97.73, 97.67, 97.67, 97.67, 97.6, 97.6, 97.67, 97.53, 97.8, 97.67, 97.67, 97.67, 97.8, 97.67, 97.67, 97.73, 97.67, 97.67, 97.67, 97.67, 97.73, 97.73, 97.8, 97.8, 97.67, 97.73, 97.73, 97.67, 97.73, 97.8, 97.6]
main_fed = [63.87, 67.2, 71.4, 74.13, 76.8, 78.8, 80.0, 80.73, 81.2, 81.87, 82.47, 82.6, 83.13, 83.47, 83.87, 84.4, 85.0, 85.07, 85.67, 85.8, 85.93, 86.07, 86.07, 86.4, 86.73, 86.53, 87.0, 87.07, 87.33, 87.2, 87.73, 87.73, 87.8, 87.73, 88.07, 87.87, 88.2, 88.27, 88.53, 88.6, 88.73, 88.73, 88.87, 89.07, 89.0, 89.13, 89.13, 89.27, 89.13, 89.0]
main_nn = [9.93, 96.4, 96.87, 97.07, 97.2, 97.2, 97.4, 97.6, 97.73, 97.53, 97.33, 97.6, 97.47, 97.47, 97.53, 97.6, 97.67, 97.6, 97.53, 97.47, 97.6, 97.53, 97.67, 97.67, 97.6, 97.53, 97.73, 97.53, 97.53, 97.53, 97.6, 97.53, 97.6, 97.6, 97.47, 97.53, 97.47, 97.6, 97.53, 97.53, 97.53, 97.6, 97.53, 97.6, 97.67, 97.53, 97.67, 97.6, 97.6, 97.6]

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
plt.ylim(85)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
