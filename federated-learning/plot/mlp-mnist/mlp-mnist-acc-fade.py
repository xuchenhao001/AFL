# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

fed_async = [13.4, 58.2, 75.4, 81.0, 83.0, 86.6, 86.4, 87.6, 89.2, 90.2, 90.2, 89.6, 90.6, 90.6, 91.0, 91.0, 91.4, 90.6, 91.0, 91.0, 91.4, 91.8, 90.8, 92.4, 92.0, 91.6, 91.8, 92.0, 91.8, 92.0, 92.0, 92.0, 91.6, 91.6, 91.8, 91.4, 91.6, 92.4, 91.4, 92.2, 91.4, 93.0, 92.0, 91.0, 91.0, 91.0, 91.0, 91.0, 91.0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
fed_async_f00 = [12.4, 53.0, 76.2, 80.4, 87.4, 89.8, 90.0, 90.8, 90.2, 90.6, 90.2, 91.4, 91.2, 91.2, 91.0, 90.8, 91.4, 91.0, 91.2, 91.4, 91.8, 91.6, 91.8, 91.6, 91.2, 91.0, 91.2, 91.0, 91.2, 91.6, 91.2, 91.0, 91.4, 91.8, 91.8, 92.0, 91.4, 91.4, 91.4, 91.2, 91.6, 92.0, 91.67, 91.5, 91.0, 91.0, 91.0, 91.0, 91.0, 91.0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
fed_async_f05 = [7.8, 56.0, 73.6, 77.4, 82.2, 86.2, 87.2, 87.0, 87.0, 88.2, 88.0, 87.6, 88.0, 88.0, 88.2, 88.0, 89.0, 88.4, 88.2, 89.2, 89.2, 89.2, 89.4, 89.4, 89.2, 89.2, 89.8, 89.6, 89.6, 90.0, 90.0, 89.8, 90.0, 90.2, 89.8, 89.8, 90.4, 90.0, 90.2, 90.4, 89.8, 90.2, 90.4, 90.8, 90.6, 90.0, 90.0, 90.0, 90.0, 90.0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
fed_async_f10 = [12.2, 51.2, 69.4, 80.4, 84.6, 86.2, 86.4, 87.0, 87.0, 86.4, 86.8, 86.4, 87.0, 87.0, 87.0, 87.0, 87.4, 87.8, 86.6, 87.6, 87.6, 87.8, 87.6, 87.6, 88.2, 88.4, 88.0, 86.8, 87.6, 87.4, 87.2, 87.8, 87.6, 86.8, 87.4, 87.8, 87.6, 87.8, 87.4, 87.6, 88.0, 88.2, 88.2, 88.0, 92.0, 91.0, 91.0, 89.0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
fed_async_f15 = [17.2, 67.2, 77.4, 79.8, 79.8, 80.4, 80.2, 80.6, 80.4, 80.6, 85.6, 87.4, 87.8, 87.8, 88.8, 88.6, 88.6, 88.4, 89.0, 89.0, 88.8, 89.4, 89.4, 88.8, 89.0, 89.0, 89.6, 89.6, 89.8, 89.6, 89.6, 90.2, 90.2, 89.4, 90.0, 90.4, 90.4, 89.6, 90.6, 90.2, 90.2, 90.2, 90.2, 90.6, 94.0, 94.0, 94.0, 94.0, 94.0, 94.0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
fed_async_f20 = [9.2, 34.0, 60.8, 70.6, 76.8, 77.2, 77.6, 78.2, 79.4, 79.6, 79.6, 81.4, 87.2, 89.4, 89.4, 90.0, 89.4, 90.0, 90.2, 90.4, 91.0, 90.8, 90.2, 90.2, 90.8, 90.6, 90.2, 90.4, 92.0, 91.2, 91.2, 91.2, 91.6, 91.6, 91.8, 91.4, 91.2, 91.0, 90.8, 90.6, 90.6, 90.8, 91.2, 91.6, 91.6, 89.33, 86.0, 85.0, 85.0, 85.0, 85.0, 85.0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

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
plt.ylim(60)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
