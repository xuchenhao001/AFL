# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async_f05 = [0.0, 13.8, 14.47, 8.47, 13.07, 39.33, 39.47, 35.13, 38.53, 36.33, 30.8, 40.67, 41.6, 40.73, 36.6, 42.6, 43.2, 43.2, 50.6, 51.0, 51.27, 51.53, 45.8, 39.73, 39.87, 38.4, 40.53, 42.8, 43.07, 41.33, 38.53, 52.6, 56.47, 54.8, 52.47, 53.07, 53.8, 53.07, 49.93, 45.53, 46.8, 45.87, 44.13, 54.67, 55.2, 55.0, 49.07, 52.8, 49.73, 59.6, 61.93, 60.33, 54.53, 59.47, 53.93, 55.53, 51.53, 47.0, 53.07, 57.67, 63.6, 60.47, 53.93, 54.33, 51.6, 53.0, 54.33, 60.2, 60.2, 56.27, 51.27, 55.0, 54.53, 58.4, 61.07, 63.8, 67.0, 66.47, 61.87, 59.2, 62.47, 62.47, 61.8, 57.33, 55.8, 57.53, 57.47, 55.07]
fed_async_f10 = [13.4, 19.0, 19.13, 13.8, 7.53, 37.13, 37.4, 36.0, 39.2, 42.0, 37.67, 44.4, 45.8, 43.07, 44.2, 50.67, 53.07, 43.6, 48.6, 43.93, 47.4, 51.33, 57.33, 53.4, 47.13, 42.87, 43.33, 43.0, 38.6, 37.67, 57.07, 57.07, 57.13, 53.2, 50.6, 58.67, 59.13, 55.07, 51.8, 51.2, 51.2, 51.33, 47.73, 50.47, 55.4, 55.13, 47.0, 53.67, 46.87, 47.0, 47.07, 48.2, 58.0, 61.13, 61.27, 55.67, 57.4, 62.33, 60.2, 58.0, 65.13, 59.93, 57.67, 54.53, 59.8, 54.07, 55.0, 53.07, 57.8, 58.0, 53.8, 56.33, 54.87, 60.87, 60.93, 56.13, 50.87, 51.6, 52.07, 51.73, 53.07, 52.33, 53.2, 60.0, 56.53, 62.33, 62.73, 61.33]
fed_async_f15 = [13.33, 26.07, 26.33, 14.27, 26.27, 34.0, 38.6, 39.47, 37.93, 37.13, 40.2, 37.13, 31.47, 30.2, 38.4, 41.4, 41.4, 43.07, 44.87, 47.93, 51.87, 44.53, 42.6, 46.4, 48.27, 44.13, 49.6, 49.6, 47.4, 47.73, 44.2, 54.93, 49.27, 55.07, 57.07, 62.6, 58.13, 53.93, 58.4, 63.47, 64.07, 56.93, 56.13, 57.87, 57.87, 53.6, 53.33, 50.13, 48.13, 62.6, 57.87, 57.87, 55.33, 53.8, 53.73, 57.07, 55.33, 61.2, 56.2, 56.0, 61.2, 50.27, 50.6, 56.6, 51.67, 52.2, 54.73, 60.8, 56.93, 50.33, 50.47, 49.67, 53.6, 53.93, 53.93, 48.2, 53.33, 53.27, 58.73, 59.87, 51.8, 51.67, 56.33, 57.33, 52.47, 48.4, 51.2, 56.6]

x = range(len(fed_async_f05))

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, fed_async_f05, label="f=0.5")
axes.plot(x, fed_async_f10, label="f=1.0 (no fade)")
axes.plot(x, fed_async_f15, label="f=1.5")
# axes.plot(x, fed_async_f20, label="f=2.0")

axes.set_xlabel("Running Time (seconds)", **csXYLabelFont)
axes.set_ylabel("Mean of Test Accuracy (%)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
# plt.ylim(40)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
