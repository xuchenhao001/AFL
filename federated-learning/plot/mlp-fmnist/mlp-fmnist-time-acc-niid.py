# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async = [6.67, 35.33, 44.67, 47.33, 49.33, 55.33, 58.67, 61.33, 67.33, 71.33, 68.0, 70.67, 72.0, 75.33, 78.0, 79.33, 82.0, 82.0]
fed_avg = [36.0, 65.33, 69.33, 73.33, 78.0, 79.33, 80.67, 80.67, 81.33, 83.33, 84.67, 84.67, 84.67, 84.67, 84.67, 84.67, 84.67, 84.67]
fed_sync = [38.0, 60.0, 70.67, 75.33, 79.33, 81.33, 82.67, 84.0, 84.67, 84.0, 84.67, 86.0, 86.67, 87.33, 87.33, 87.33, 88.67, 88.67]
fed_localA = [24.0, 67.33, 67.33, 58.67, 73.33, 80.67, 68.0, 90.67, 58.89, None, None, None, None, None, None, None, None, None]
local_train = [77.33, 65.56, 100.0, 100.0, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

x = range(len(fed_async))
x = [value * 10 for value in x]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, fed_async, label="AFL", linewidth=3)
axes.plot(x, fed_avg, label="FedAVG", linestyle='--', alpha=0.5)
axes.plot(x, fed_sync, label="SFL", linestyle='--', alpha=0.5)
axes.plot(x, fed_localA, label="APFL", linestyle='--', alpha=0.5)
axes.plot(x, local_train, label="Local Training", linestyle='--', alpha=0.5)

axes.set_xlabel("Running Time (seconds)", **csXYLabelFont)
axes.set_ylabel("Mean of Test Accuracy (%)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
# plt.ylim(40)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
