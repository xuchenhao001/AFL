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
fed_server = [48.73, 54.93, 59.93, 65.0, 68.6, 69.73, 70.27, 70.07, 71.67, 71.4, 71.47, 72.2, 72.87, 73.53, 72.93, 73.07, 73.73, 72.6, 73.07, 72.8, 72.67, 72.73, 72.87, 73.0, 73.07, 72.73, 73.2, 72.6, 72.53, 72.47, 72.93, 72.53, 73.4, 73.4, 72.53, 72.8, 72.4, 72.67, 72.4, 73.0, 72.67, 72.87, 73.27, 73.07, 73.0, 72.8, 73.27, 72.47, 71.93, 71.73]
main_fed_localA = [6.93, 43.0, 58.27, 63.27, 65.6, 69.6, 71.8, 72.93, 72.33, 72.07, 72.8, 62.33, 72.93, 72.8, 73.93, 73.93, 73.8, 73.47, 73.2, 73.13, 73.4, 73.47, 73.27, 73.2, 73.47, 73.47, 73.53, 73.53, 73.4, 73.4, 73.4, 73.4, 73.33, 73.27, 73.53, 73.47, 73.47, 73.53, 73.33, 73.33, 73.27, 73.2, 73.33, 73.47, 73.27, 73.33, 73.27, 73.27, 73.13, 73.13]
main_fed = [22.67, 29.67, 32.0, 34.13, 37.93, 40.07, 40.47, 41.8, 42.4, 42.87, 43.87, 44.47, 45.2, 46.73, 46.2, 46.6, 47.27, 47.93, 47.67, 47.2, 47.0, 47.67, 48.67, 48.27, 48.27, 47.47, 48.6, 48.27, 47.87, 48.6, 48.27, 48.2, 47.07, 47.87, 48.53, 47.53, 48.27, 47.93, 49.0, 48.33, 49.67, 49.47, 48.4, 49.07, 47.87, 48.47, 48.53, 49.13, 49.27, 49.73]
main_nn = [6.93, 49.93, 57.13, 63.47, 66.6, 68.0, 70.87, 70.53, 70.33, 71.47, 72.87, 72.4, 72.33, 72.4, 72.33, 72.4, 72.6, 72.6, 72.53, 72.4, 72.33, 72.4, 72.4, 72.33, 72.27, 72.33, 72.27, 72.27, 72.33, 72.27, 72.33, 72.33, 72.4, 72.47, 72.4, 72.4, 72.4, 72.4, 72.4, 72.4, 72.4, 72.4, 72.53, 72.4, 72.4, 72.4, 72.4, 72.4, 72.47, 72.47]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, fed_server, label="SCEI with negotiated α", linewidth=3)
axes.plot(x, main_nn, label="Local Training", linestyle='--', alpha=0.5)
axes.plot(x, main_fed_localA, label="APFL", linestyle='--', alpha=0.5)
axes.plot(x, main_fed, label="FedAvg", linestyle='--', alpha=0.5)

axes.set_xlabel("Training Rounds", **csXYLabelFont)
axes.set_ylabel("Mean of Local Test Accuracy (%)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
plt.ylim(40)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
