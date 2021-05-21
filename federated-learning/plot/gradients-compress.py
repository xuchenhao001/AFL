# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

x = ["CNN-CIFAR10", "CNN-MNIST", "MLP-MNIST"]

before_comp = [2610.831055, 1003.351563, 2103.550781]
after_comp = [780.1748047, 279.9521484, 639.3232422]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

width = 0.15  # the width of the bars
axes.bar([p - width/2 for p in range(len(x))], height=before_comp, width=width, label="Before compress", hatch='x')
axes.bar([p + width/2 for p in range(len(x))], height=after_comp, width=width, label="After compress", hatch='*')

plt.xticks(range(len(x)), x)
axes.set_xlabel("The Type of Model and Datasets", **csXYLabelFont)
axes.set_ylabel("The Size of Gradients (KB)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
plt.tight_layout()
# plt.ylim(90, 100)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
