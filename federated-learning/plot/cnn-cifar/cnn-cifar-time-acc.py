# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async = [10.0, 12.0, 19.0, 22.2, 28.0, 29.8, 32.4, 32.2, 35.6, 33.0, 33.4, 34.8, 35.0, 36.2, 36.8, 36.2, 35.6, 36.2, 36.0, 36.6, 36.0, 35.4, 36.4, 35.0, 36.6, 35.6, 35.8, 37.0, 35.6, 35.6, 34.0, 34.4, 35.6, 35.6, 34.6, 34.8, 37.2, 36.8, 35.4, 35.4, 36.2, 35.2, 35.6, 35.0, 35.4]
fed_sync = [10.0, 10.2, 15.6, 15.6, 16.8, 21.6, 24.0, 24.0, 28.8, 29.4, 31.0, 32.4, 32.4, 32.0, 35.0, 35.0, 35.0, 35.8, 37.4, 37.6, 36.4, 37.2, 37.0, 37.0, 37.8, 39.0, 37.4, 37.4, 36.6, 37.2, 37.4, 37.4, 36.4, 35.8, 35.8, 35.6, 35.0, 35.6, 36.6, 34.8, 33.4, 33.4, 34.4, 34.4, 34.4]
fed_localA = [10.4, 10.4, 10.4, 10.4, 12.2, 16.6, 19.4, 19.6, 21.8, 22.4, 25.2, 27.0, 24.6, 25.0, 25.6, 28.0, 29.2, 27.6, 26.2, 24.0, 24.2, 25.8, 24.8, 24.8, 24.2, 22.8, 23.6, 27.8, 31.0, 29.4, 28.4, 28.2, 29.4, 30.2, 30.8, 30.2, 30.6, 30.2, 30.2, 30.0, 30.2, 31.2, 30.6, 29.8, 30.2]
local_train = [10.2, 11.2, 14.2, 16.4, 15.8, 20.6, 22.8, 22.8, 26.2, 24.0, 25.4, 24.0, 27.4, 24.6, 25.6, 28.2, 27.2, 27.8, 27.6, 25.6, 28.4, 25.4, 25.0, 25.0, 25.2, 25.4, 27.0, 25.4, 28.4, 25.4, 25.4, 25.2, 25.2, 25.2, 23.6, 25.2, 25.2, 23.6, 24.8, 24.8, 25.2, 25.0, 25.2, 25.0, 24.8]

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
# plt.ylim(40)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
