# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async = [10.0, 11.2, 16.0, 17.8, 21.2, 22.4, 27.4, 29.8, 34.6, 34.6, 38.2, 38.6, 40.4, 41.8, 40.8, 39.6, 42.0, 41.8, 42.6, 42.0, 43.4, 43.0, 44.8, 44.2, 45.0, 44.8, 43.8, 43.2, 44.4, 43.8, 44.4, 44.2, 44.6, 46.8, 44.0, 43.8, 43.4, 43.4, 44.4, 44.8, 44.8, 43.8, 43.8, 48.33, 48.0, 49.67, 49.67, 49.67, 49.67, 50.0, 49.67, 49.67, 49.0, 48.0, 48.0, 48.0, 48.0, 47.0, 48.0, 48.0, 48.0, 48.0, 47.0, 47.0, 47.0, 47.0, 47.0, 47.0, 47.0, 47.0, 47.0, 47.0, 47.0, 47.0, 46.0, 47.0, 47.0, 48.0]
fed_avg = [9.8, 10.0, 10.6, 10.2, 10.6, 13.0, 13.0, 17.4, 17.4, 17.8, 19.4, 23.2, 27.8, 28.4, 29.8, 31.8, 34.4, 35.2, 36.2, 40.4, 40.4, 41.0, 40.6, 42.0, 44.6, 45.0, 42.6, 42.6, 42.8, 42.6, 42.0, 41.6, 41.6, 43.4, 43.4, 42.6, 43.0, 42.8, 42.2, 41.6, 42.0, 42.0, 42.8, 42.6, 41.4, 41.6, 41.6, 41.2, 41.4, 42.6, 42.8, 41.6, 41.0, 41.0, 42.2, 42.4, 41.6, 41.8, 42.0, 41.0, 39.6, 40.8, 41.4, 39.2, 39.2, 39.8, 40.2, 40.6, 39.8, 39.8, 40.0, 39.2, 39.2, 39.6, 39.6, 39.2, 38.4, 39.2]
fed_sync = [10.0, 10.2, 15.6, 15.6, 16.8, 21.6, 24.0, 24.0, 28.8, 29.4, 31.0, 32.4, 32.4, 32.0, 35.0, 35.0, 35.0, 35.8, 37.4, 37.6, 36.4, 37.2, 37.0, 37.0, 37.8, 39.0, 37.4, 37.4, 36.6, 37.2, 37.4, 37.4, 36.4, 35.8, 35.8, 35.6, 35.0, 35.6, 36.6, 34.8, 33.4, 33.4, 34.4, 34.4, 34.4, 34.4, 33.8, 34.8, 34.8, 33.8, 33.4, 33.4, 33.2, 33.0, 33.0, 33.0, 33.0, 33.0, 33.4, 33.2, 33.2, 33.4, 33.2, 33.6, 33.6, 33.4, 33.4, 33.4, 33.2, None, None, None, None, None, None, None, None, None]
fed_localA = [10.4, 10.4, 10.4, 10.4, 12.2, 16.6, 19.4, 19.6, 21.8, 22.4, 25.2, 27.0, 24.6, 25.0, 25.6, 28.0, 29.2, 27.6, 26.2, 24.0, 24.2, 25.8, 24.8, 24.8, 24.2, 22.8, 23.6, 27.8, 31.0, 29.4, 28.4, 28.2, 29.4, 30.2, 30.8, 30.2, 30.6, 30.2, 30.2, 30.0, 30.2, 31.2, 30.6, 29.8, 30.2, 30.0, 30.0, 29.2, 29.2, 29.2, 30.2, 32.2, 32.4, 31.6, 30.6, 30.8, 31.4, 31.4, 30.8, 30.2, 30.2, 29.8, 29.8, 30.4, 30.0, 29.8, 30.4, 30.4, 30.2, 29.8, 29.8, 29.8, 29.8, 29.8, 32.0, 32.6, 29.2, 28.8]
local_train = [10.2, 11.2, 14.2, 16.4, 15.8, 20.6, 22.8, 22.8, 26.2, 24.0, 25.4, 24.0, 27.4, 24.6, 25.6, 28.2, 27.2, 27.8, 27.6, 25.6, 28.4, 25.4, 25.0, 25.0, 25.2, 25.4, 27.0, 25.4, 28.4, 25.4, 25.4, 25.2, 25.2, 25.2, 23.6, 25.2, 25.2, 23.6, 24.8, 24.8, 25.2, 25.0, 25.2, 25.0, 24.8, 25.0, 25.0, 25.2, 25.2, 26.0, 26.0, 26.33, 26.0, 25.0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

x = range(len(fed_async))
x = [value * 10 for value in x]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=19)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, fed_async, label="BAFL", linewidth=3)
axes.plot(x, fed_sync, label="BSFL", linestyle='--', alpha=0.5)
axes.plot(x, fed_avg, label="FedAVG", linestyle='--', alpha=0.5)
axes.plot(x, fed_localA, label="APFL", linestyle='--', alpha=0.5)
axes.plot(x, local_train, label="Local Training", linestyle='--', alpha=0.5)

axes.set_xlabel("Running Time (seconds)", **csXYLabelFont)
axes.set_ylabel("Mean of Test Accuracy (%)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
plt.tight_layout()
# plt.ylim(40)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
