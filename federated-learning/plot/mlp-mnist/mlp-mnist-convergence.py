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
network_5_node = [94.67, 95.6, 96.67, 96.8, 96.53, 96.93, 96.53, 96.53, 96.53, 96.67, 96.53, 96.67, 96.8, 97.07, 96.93, 97.07, 97.2, 97.07, 97.33, 97.07, 97.07, 97.2, 97.6, 96.93, 97.2, 97.07, 97.2, 97.2, 97.33, 97.33, 97.47, 97.6, 97.6, 97.47, 97.47, 97.73, 97.73, 97.6, 97.73, 97.6, 97.87, 97.6, 97.87, 98.0, 97.87, 98.0, 98.0, 97.87, 98.13, 97.6]
network_10_node = [95.87, 96.8, 97.07, 97.33, 97.4, 97.4, 97.33, 97.07, 97.47, 97.2, 97.2, 97.2, 97.33, 97.33, 97.4, 97.4, 97.4, 97.27, 97.27, 97.33, 97.4, 97.33, 97.4, 97.4, 97.47, 97.27, 97.4, 97.4, 97.33, 97.47, 97.53, 97.4, 97.67, 97.47, 97.4, 97.67, 97.47, 97.6, 97.67, 97.53, 97.53, 97.47, 97.53, 97.47, 97.67, 97.53, 97.67, 97.47, 97.6, 97.73]
network_20_node = [62.37, 65.4, 72.83, 74.97, 75.83, 78.7, 81.47, 85.03, 85.73, 84.1, 84.9, 86.13, 85.47, 85.23, 86.17, 87.73, 87.67, 86.77, 86.0, 87.8, 87.93, 88.4, 90.17, 90.33, 88.47, 88.13, 88.47, 91.2, 88.23, 87.8, 88.87, 90.33, 91.27, 89.13, 89.0, 89.33, 89.23, 88.07, 90.27, 90.67, 90.6, 90.1, 88.8, 89.9, 90.87, 89.87, 91.3, 90.23, 91.7, 89.1]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, network_5_node, label="SCEI with 5 nodes")
axes.plot(x, network_10_node, label="SCEI with 10 nodes")
axes.plot(x, network_20_node, label="SCEI with 20 nodes")

axes.set_xlabel("Training Rounds", **csXYLabelFont)
axes.set_ylabel("Mean of Local Test Accuracy (%)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
plt.ylim(80)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
