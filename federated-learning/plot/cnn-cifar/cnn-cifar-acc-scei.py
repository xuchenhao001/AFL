# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
fed_server = [48.73, 54.93, 59.93, 65.0, 68.6, 69.73, 70.27, 70.07, 71.67, 71.4, 71.47, 72.2, 72.87, 73.53, 72.93, 73.07, 73.73, 72.6, 73.07, 72.8, 72.67, 72.73, 72.87, 73.0, 73.07, 72.73, 73.2, 72.6, 72.53, 72.47, 72.93, 72.53, 73.4, 73.4, 72.53, 72.8, 72.4, 72.67, 72.4, 73.0, 72.67, 72.87, 73.27, 73.07, 73.0, 72.8, 73.27, 72.47, 71.93, 71.73]
fed_server_alpha_025 = [40.53, 45.87, 45.2, 47.8, 48.8, 52.53, 54.33, 56.07, 57.27, 58.47, 59.0, 59.53, 60.67, 60.47, 62.07, 62.8, 61.8, 63.4, 63.33, 62.2, 63.4, 64.0, 64.93, 64.2, 65.53, 65.2, 63.27, 63.87, 64.0, 63.8, 64.6, 64.73, 63.73, 63.0, 64.27, 63.13, 62.8, 62.47, 63.47, 62.47, 64.07, 62.87, 62.67, 62.47, 63.93, 62.13, 62.13, 62.93, 63.4, 62.67]
fed_server_alpha_050 = [44.53, 52.53, 56.47, 57.33, 62.13, 63.4, 64.8, 66.0, 66.6, 68.33, 68.07, 69.8, 69.73, 69.67, 69.93, 70.07, 69.87, 69.2, 69.93, 69.87, 69.2, 69.93, 69.2, 70.6, 71.07, 70.4, 70.27, 70.0, 70.53, 69.47, 69.4, 69.93, 70.27, 71.07, 70.6, 69.73, 69.87, 69.4, 69.53, 70.73, 69.0, 69.47, 70.33, 70.87, 69.67, 68.6, 69.13, 69.4, 69.4, 69.07]
fed_server_alpha_075 = [48.6, 57.6, 60.67, 63.87, 67.47, 68.67, 69.13, 69.8, 71.2, 72.13, 70.2, 72.47, 72.2, 72.73, 72.2, 73.13, 73.0, 72.73, 72.2, 71.73, 72.13, 72.33, 72.13, 72.33, 71.87, 72.27, 72.47, 72.87, 72.33, 72.8, 72.6, 72.27, 73.0, 72.73, 72.33, 73.07, 73.0, 72.73, 72.67, 71.47, 72.4, 72.93, 72.73, 72.73, 72.93, 72.6, 71.73, 72.33, 72.53, 72.4]
main_fed = [22.67, 29.67, 32.0, 34.13, 37.93, 40.07, 40.47, 41.8, 42.4, 42.87, 43.87, 44.47, 45.2, 46.73, 46.2, 46.6, 47.27, 47.93, 47.67, 47.2, 47.0, 47.67, 48.67, 48.27, 48.27, 47.47, 48.6, 48.27, 47.87, 48.6, 48.27, 48.2, 47.07, 47.87, 48.53, 47.53, 48.27, 47.93, 49.0, 48.33, 49.67, 49.47, 48.4, 49.07, 47.87, 48.47, 48.53, 49.13, 49.27, 49.73]
main_nn = [6.93, 49.93, 57.13, 63.47, 66.6, 68.0, 70.87, 70.53, 70.33, 71.47, 72.87, 72.4, 72.33, 72.4, 72.33, 72.4, 72.6, 72.6, 72.53, 72.4, 72.33, 72.4, 72.4, 72.33, 72.27, 72.33, 72.27, 72.27, 72.33, 72.27, 72.33, 72.33, 72.4, 72.47, 72.4, 72.4, 72.4, 72.4, 72.4, 72.4, 72.4, 72.4, 72.53, 72.4, 72.4, 72.4, 72.4, 72.4, 72.47, 72.47]

fig, axes = plt.subplots()

legendFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
xylabelFont = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
csXYLabelFont = {'fontproperties': xylabelFont}

axes.plot(x, fed_server, label="negotiated α (0.5-0.8)", linewidth=3)
axes.plot(x, main_fed, label="α=0.0 (i.e. FedAvg)", linestyle='--', alpha=0.5)
axes.plot(x, fed_server_alpha_025, label="α=0.25", linestyle='--', alpha=0.5)
axes.plot(x, fed_server_alpha_050, label="α=0.5", linestyle='--', alpha=0.5)
axes.plot(x, fed_server_alpha_075, label="α=0.75", linestyle='--', alpha=0.5)
axes.plot(x, main_nn, label="α=1.0 (i.e. Local Training)", linestyle='--', alpha=0.5)


axes.set_xlabel("Training Rounds", **csXYLabelFont)
axes.set_ylabel("Mean of Local Test Accuracy (%)", **csXYLabelFont)

plt.xticks(family='Times New Roman', fontsize=15)
plt.yticks(family='Times New Roman', fontsize=15)
plt.ylim(40)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
