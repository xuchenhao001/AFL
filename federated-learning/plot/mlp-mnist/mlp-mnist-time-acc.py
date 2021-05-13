# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async = [7.33, 24.33, 24.33, 15.67, 47.2, 47.2, 54.93, 48.47, 69.33, 89.67, 83.27, 79.87, 87.47, 82.47, 78.8, 84.33, 83.4, 82.33, 82.2, 84.33, 85.13, 82.4, 76.13, 78.33, 79.6, 83.6, 85.07, 86.93, 85.27, 83.67]
fed_sync = [1.07, 1.07, 63.87, 63.87, 73.8, 73.8, 75.73, 78.8, 78.8, 80.93, 80.93, 82.47, 82.47, 82.73, 84.0, 84.0, 85.0, 85.0, 85.33, 86.4, 86.4, 86.87, 86.87, 86.87, 87.2, 87.2, 87.2, 87.67, 87.67, 87.8]
fed_localA = [12.07, 12.07, 12.07, 12.07, 12.07, 12.07, 12.07, 45.33, 92.47, 92.47, 94.27, 97.93, 98.33, 98.47, 98.53, 98.53, 98.53, 98.47, 98.47, 98.47, 98.4, 98.33, 98.4, 98.4, 98.27, 98.27, 98.27, 98.27, 98.27, 98.2]
local_train = [28.0, 28.13, 28.27, 95.53, 95.73, 95.93, 96.47, 96.07, 96.07, 96.6, 96.53, 96.4, 96.4, 96.4, 96.47, 96.33, 96.33, 96.4, 96.53, 96.33, 96.6, 97.0, 96.93, 96.73, 96.67, 96.73, 96.87, 96.53, 96.53, 96.73]

x = range(len(fed_async))

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, fed_async, label="AFL", linewidth=3)
axes.plot(x, fed_sync, label="SFL", linestyle='--', alpha=0.5)
axes.plot(x, fed_localA, label="APFL", linestyle='--', alpha=0.5)
axes.plot(x, local_train, label="Local Training", linestyle='--', alpha=0.5)

axes.set_xlabel("Running Time (seconds)", **csXYLabelFont)
axes.set_ylabel("Mean of Test Accuracy (%)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
# plt.ylim(85)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
