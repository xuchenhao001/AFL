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
network_5_node = [94.93, 97.73, 97.87, 98.13, 98.27, 98.93, 98.4, 98.27, 98.27, 98.53, 98.67, 98.8, 98.4, 98.4, 98.53, 98.4, 99.07, 98.93, 98.93, 98.67, 98.93, 98.67, 98.4, 98.53, 98.93, 98.67, 98.67, 98.67, 98.8, 98.53, 98.93, 98.8, 98.93, 98.93, 98.8, 98.67, 99.07, 99.07, 99.07, 98.8, 99.2, 98.93, 99.33, 98.93, 98.93, 99.2, 99.33, 99.2, 98.93, 99.2]
network_10_node = [96.87, 97.8, 98.33, 98.27, 98.67, 98.87, 98.6, 98.93, 99.13, 98.87, 98.87, 98.8, 99.07, 99.07, 98.93, 99.07, 99.07, 99.2, 99.07, 99.27, 99.2, 99.07, 99.0, 99.07, 99.13, 99.13, 99.0, 99.13, 99.27, 99.07, 99.2, 99.07, 99.33, 99.47, 99.2, 99.47, 99.2, 99.4, 99.47, 99.53, 99.4, 99.2, 99.33, 99.53, 99.47, 99.47, 99.6, 99.27, 99.47, 99.27]
network_20_node = [62.9, 64.23, 67.57, 73.53, 80.2, 86.17, 87.77, 84.73, 90.47, 91.2, 91.07, 90.03, 88.77, 92.2, 91.33, 90.17, 91.33, 92.93, 91.7, 91.77, 91.7, 95.77, 92.67, 91.13, 92.33, 94.47, 95.93, 92.77, 94.03, 94.6, 93.53, 92.6, 93.0, 96.43, 94.07, 94.27, 93.97, 96.7, 94.83, 94.4, 94.4, 97.4, 95.63, 95.3, 94.23, 97.43, 95.4, 94.33, 95.43, 94.9]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, network_5_node, label="SCEI with 5 nodes")
axes.plot(x, network_10_node, label="SCEI with 10 nodes")
axes.plot(x, network_20_node, label="SCEI with 20 nodes")
# axes.plot(x, network_50_peer, label="SCEI with 50 nodes")

axes.set_xlabel("Training Rounds", **csXYLabelFont)
axes.set_ylabel("Mean of Local Test Accuracy (%)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
plt.ylim(85)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
