# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async = [37.8, 81.6, 84.8, 89.8, 90.4, 91.6, 92.0, 93.0, 92.6, 94.4, 94.2, 94.0, 93.8, 94.4, 94.8, 94.6, 94.2]
fed_sync = [34.6, 87.2, 90.8, 92.6, 93.0, 92.8, 94.4, 94.4, 93.6, 94.8, 94.4, 94.8, 94.6, 94.2, 95.0, 95.2, 95.6]
fed_localA = [10.0, 10.0, 62.8, 79.6, 89.2, 89.6, 90.2, 91.0, 93.2, 93.6, 93.6, 93.2, 92.8, 92.8, 92.8, 93.0, 91.8]
local_train = [57.4, 85.2, 88.4, 91.6, 92.2, 92.2, 93.2, 92.0, 91.4, 93.8, 94.4, 93.4, 93.4, 93.4, 93.2, 92.8, 92.67]

x = range(len(fed_async))
x = [value * 10 for value in x]

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
plt.ylim(20)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
