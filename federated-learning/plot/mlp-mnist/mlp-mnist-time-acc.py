# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async = [17.4, 60.8, 81.2, 85.4, 85.4, 85.8, 87.4, 87.4, 87.0, 87.4, 87.8, 88.6, 89.2, 88.6, 88.4, 88.0, 88.0, 88.2]
fed_avg = [19.6, 59.4, 73.4, 77.6, 78.4, 81.8, 84.6, 86.8, 87.6, 88.4, 88.4, 88.4, 88.4, 89.2, 88.6, 88.0, 88.6, 89.4]
fed_sync = [50.0, 67.0, 71.8, 73.2, 80.0, 80.2, 81.8, 81.8, 82.2, 82.0, 82.2, 82.8, 82.8, 82.4, 82.6, 84.8, 89.2, 90.2]
fed_localA = [9.4, 68.2, 77.2, 79.8, 81.2, 81.2, 85.8, 86.2, 85.8, 86.0, 86.0, 86.4, 85.4, 86.4, 86.0, 86.0, 86.4, 87.6]
local_train = [67.2, 79.2, 81.4, 78.6, 81.0, 85.0, 84.0, 83.0, 84.0, 83.0, 84.0, None, None, None, None, None, None, None]

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
plt.ylim(50)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
