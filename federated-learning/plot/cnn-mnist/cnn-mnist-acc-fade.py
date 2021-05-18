# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async = [44.2, 86.4, 88.2, 91.0, 92.8, 94.4, 95.8, 94.6, 96.0, 96.2, 96.4, 96.4, 96.8, 96.2, 96.8, 97.4, 96.6, 97.0, 97.8, 97.6, 97.8, 97.4, 97.0, 97.4, 98.2, 98.0, 98.0, 98.2, 98.2, 98.4, 98.0, 97.8, 97.6, 97.8, 98.2, 97.6, 98.4, 97.67, 97.67, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.0, 100.0, 99.0, 99.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.0, 98.0, 98.0, 98.0, 99.0, 99.0, None, None, None, None, None, None, None, None, None, None, None]
fed_async_f00 = [34.4, 84.8, 88.0, 89.6, 93.0, 93.0, 93.0, 94.6, 94.4, 95.0, 95.6, 96.0, 95.6, 96.2, 96.2, 96.8, 97.0, 96.0, 96.2, 96.4, 96.6, 96.8, 96.2, 96.6, 96.8, 96.6, 96.8, 96.6, 96.6, 97.2, 96.6, 96.8, 97.0, 96.8, 96.4, 97.0, 96.2, 97.0, 96.6, 96.6, 97.2, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, None, None, None, None, None, None, None]
fed_async_f05 = [60.4, 82.2, 86.0, 90.8, 91.0, 93.0, 93.0, 93.6, 92.8, 94.0, 93.6, 93.6, 93.2, 94.6, 94.2, 94.0, 94.8, 94.4, 94.8, 95.4, 94.8, 95.0, 95.0, 95.4, 95.4, 95.6, 95.8, 96.0, 95.6, 95.4, 95.6, 96.0, 95.8, 96.2, 96.4, 96.4, 96.2, 96.2, 96.0, 96.2, 96.2, 96.0, 96.0, 97.0, 96.0, 95.0, 96.0, 96.0, 96.0, 96.0, 96.0, 95.0, 95.0, 95.0, 95.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 95.0, 95.0, 95.0, 96.0, 95.0, 95.0, None, None, None, None, None, None, None, None, None]
fed_async_f10 = [42.2, 87.8, 89.4, 91.6, 91.8, 93.2, 94.2, 94.8, 95.4, 95.4, 95.6, 95.6, 95.8, 96.0, 95.8, 95.8, 95.8, 96.0, 95.8, 96.0, 96.2, 96.4, 95.6, 96.2, 95.8, 96.2, 95.6, 95.8, 96.6, 96.4, 96.2, 96.0, 96.2, 95.6, 95.8, 95.6, 96.2, 96.2, 96.2, 96.4, 96.0, 98.0, 98.0, 98.0, 98.0, 97.0, 97.0, 97.0, 97.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 97.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, None, None, None, None, None, None, None, None, None]
fed_async_f15 = [33.8, 85.4, 88.8, 91.6, 94.8, 94.2, 94.6, 95.0, 94.2, 95.2, 96.4, 95.8, 96.6, 96.8, 97.2, 97.0, 97.2, 97.4, 96.8, 96.6, 97.0, 97.0, 96.6, 97.2, 96.6, 97.2, 96.8, 96.6, 96.6, 97.2, 97.0, 97.0, 97.0, 97.4, 97.2, 97.4, 97.4, 97.25, 98.33, 97.67, 98.0, 98.0, 97.0, 97.0, 98.0, 98.0, 98.0, 98.0, 98.0, 97.0, 96.0, 97.0, 97.0, 97.0, 97.0, 98.0, 97.0, 97.0, 97.0, 97.0, 97.0, 98.0, 98.0, 98.0, 97.0, 97.0, 97.0, 97.0, None, None, None, None, None, None, None, None, None, None]
fed_async_f20 = [28.0, 84.0, 84.8, 89.6, 91.0, 93.6, 92.0, 93.2, 93.4, 93.4, 94.2, 94.6, 94.0, 94.6, 94.6, 94.2, 94.6, 94.2, 94.6, 95.2, 95.2, 94.8, 94.8, 95.0, 95.0, 94.6, 94.8, 94.8, 95.4, 94.4, 95.0, 94.2, 94.4, 94.2, 94.6, 94.4, 95.4, 94.6, 94.6, 95.2, 95.4, 94.8, 97.0, 97.0, 97.0, 96.0, 95.0, 96.0, 96.0, 95.0, 96.0, 97.0, 97.0, 97.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 96.0, 96.0, 95.0, 95.0, 95.0, 95.0, 96.0, None, None, None, None, None, None, None, None, None]

x = range(len(fed_async_f05))
x = [value * 10 for value in x]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=19)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, fed_async, label="Dynamic", linewidth=3)
axes.plot(x, fed_async_f00, label="f=0.0", linestyle='--', alpha=0.5)
axes.plot(x, fed_async_f05, label="f=0.5", linestyle='--', alpha=0.5)
axes.plot(x, fed_async_f10, label="f=1.0 (no fade)", linestyle='--', alpha=0.5)
axes.plot(x, fed_async_f15, label="f=1.5", linestyle='--', alpha=0.5)
axes.plot(x, fed_async_f20, label="f=2.0", linestyle='--', alpha=0.5)

axes.set_xlabel("Running Time (seconds)", **csXYLabelFont)
axes.set_ylabel("Mean of Test Accuracy (%)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
plt.tight_layout()
plt.ylim(80)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
