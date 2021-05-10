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
network_5_node = [52.8, 55.33, 57.73, 62.8, 66.4, 67.47, 69.2, 67.47, 68.53, 67.6, 69.33, 68.13, 67.6, 69.87, 69.2, 68.93, 69.33, 68.93, 69.07, 69.47, 69.07, 69.2, 69.47, 69.07, 69.07, 69.33, 68.67, 68.13, 68.4, 68.53, 67.87, 68.53, 68.13, 68.27, 68.27, 68.0, 68.0, 69.07, 68.67, 69.07, 67.33, 68.0, 68.67, 68.67, 68.67, 68.4, 68.8, 68.67, 68.27, 68.53]
network_10_node = [48.73, 54.93, 59.93, 65.0, 68.6, 69.73, 70.27, 70.07, 71.67, 71.4, 71.47, 72.2, 72.87, 73.53, 72.93, 73.07, 73.73, 72.6, 73.07, 72.8, 72.67, 72.73, 72.87, 73.0, 73.07, 72.73, 73.2, 72.6, 72.53, 72.47, 72.93, 72.53, 73.4, 73.4, 72.53, 72.8, 72.4, 72.67, 72.4, 73.0, 72.67, 72.87, 73.27, 73.07, 73.0, 72.8, 73.27, 72.47, 71.93, 71.73]
network_20_node = [31.73, 37.6, 37.23, 39.97, 43.97, 46.93, 44.5, 45.33, 47.2, 45.83, 45.57, 45.83, 47.2, 50.97, 50.0, 49.3, 50.07, 50.23, 51.07, 49.77, 51.5, 52.27, 52.9, 52.27, 52.7, 51.97, 52.07, 52.47, 52.63, 52.9, 52.0, 51.9, 53.7, 53.7, 52.43, 52.4, 52.47, 53.27, 52.77, 52.67, 53.07, 53.3, 52.9, 52.53, 53.37, 53.33, 53.47, 53.17, 54.17, 54.63]

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
# plt.ylim(40)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
