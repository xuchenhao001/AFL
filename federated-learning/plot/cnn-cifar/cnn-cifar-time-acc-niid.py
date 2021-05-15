# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async = [6.67, 13.33, 12.67, 12.67, 16.0, 20.0, 22.0, 10.67, 14.67, 20.67, 22.67, 14.0, 20.0, 14.67, 11.33, 25.33, 31.33, 27.33, 24.67, 27.33, 34.67, 12.0, 16.67, 18.0, 16.0, 25.33, 15.33, 12.0, 24.0, 22.0, 18.67, 20.0, 30.67, 30.0, 25.33, 23.33, 14.67, 21.33, 22.0, 17.33, 26.67, 39.17, 55.56, 52.22, 63.33, 63.33]
fed_avg = [16.67, 22.0, 21.33, 28.67, 25.33, 26.0, 28.67, 29.33, 32.67, 32.0, 29.33, 38.0, 34.67, 38.0, 38.0, 40.67, 41.33, 41.33, 40.0, 39.33, 40.0, 40.0, 37.33, 42.0, 40.67, 38.0, 37.33, 37.33, 41.33, 38.0, 40.67, 42.0, 36.67, 34.67, 38.0, 37.33, 34.0, 41.33, 32.0, 33.33, 37.33, 32.0, 33.33, 32.0, 31.67, None]
fed_sync = [22.67, 20.67, 24.0, 24.0, 24.0, 26.67, 26.67, 27.33, 28.67, 32.0, 27.33, 32.67, 32.0, 32.67, 30.0, 32.0, 34.67, 29.33, 32.0, 31.33, 32.0, 34.0, 34.67, 36.0, 34.0, 32.0, 32.67, 34.0, 38.0, 36.67, 30.67, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
fed_localA = [12.0, 18.0, 26.67, 24.67, 29.33, 34.67, 47.33, 48.0, 44.67, 40.0, 34.67, 42.67, 40.0, 44.0, 51.33, 58.0, 51.33, 55.33, 44.67, 41.33, 44.0, 44.0, 50.67, 44.67, 52.0, 42.67, 48.0, 42.67, 50.67, 60.0, 47.33, 48.67, 46.0, 41.33, 52.22, None, None, None, None, None, None, None, None, None, None, None]
local_train = [13.33, 30.67, 36.0, 26.67, 43.33, 34.67, 26.67, 22.67, 24.0, 32.0, 24.0, 28.67, 50.0, 24.0, 33.33, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

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
