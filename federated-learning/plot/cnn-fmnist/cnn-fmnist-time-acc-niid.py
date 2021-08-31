# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async = [20.0, 31.33, 52.0, 55.33, 59.33, 61.33, 56.67, 58.0, 66.0, 68.0, 70.67, 70.0, 80.67, 80.67, 84.0, 82.0, 74.0, 74.67]
fed_avg = [37.33, 60.0, 68.0, 74.0, 77.33, 80.67, 82.67, 82.0, 88.0, 84.0, 85.33, 86.67, 90.67, 89.33, 91.33, 90.0, 88.67, 92.0]
fed_sync = [49.33, 63.33, 80.0, 86.0, 90.0, 90.67, 91.33, 93.33, 92.67, 92.67, 92.67, 94.0, 92.67, 94.67, 94.67, 94.67, 96.0, 93.33]
fed_localA = [10.67, 56.67, 56.0, 80.0, 80.67, 80.67, 84.0, 85.33, 97.33, 97.33, 84.67, 84.0, 90.0, 90.0, 54.0, 90.67, 97.33, 97.33]
local_train = [64.0, 84.0, 90.67, 78.67, 86.67, 76.67, 100.0, 100.0, 100.0, 100.0, None, None, None, None, None, None, None, None]

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
