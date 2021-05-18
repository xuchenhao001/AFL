# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

x = ["Train", "Test", "Communication", "Round"]

fed_async = [2.64, 0.05, 8.59, 11.28]
fed_avg = [2.75, 0.04, 8.12, 10.91]
fed_localA = [21.12, 0.09, 2.42, 23.63]
fed_sync = [2.79, 0.05, 10.27, 13.12]
local_train = [10.31, 0.12, 0.0, 10.43]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

width = 0.15  # the width of the bars
axes.bar([(p - width * 2) for p in range(len(x))], height=fed_async, width=width, label="BAFL", hatch='x')
axes.bar([p - width for p in range(len(x))], height=fed_sync, width=width, label="BSFL", hatch='o')
axes.bar(range(len(x)), height=fed_avg, width=width, label="FedAVG", hatch='+')
axes.bar([p + width for p in range(len(x))], height=fed_localA, width=width, label="APFL", hatch='*')
axes.bar([(p + width * 2) for p in range(len(x))], height=local_train, width=width, label="Local Training", hatch='/')

plt.xticks(range(len(x)), x)
axes.set_xlabel("The Type of Time Cost", **csXYLabelFont)
axes.set_ylabel("Average Time Per Iteration (s)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
plt.tight_layout()
# plt.ylim(90, 100)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
