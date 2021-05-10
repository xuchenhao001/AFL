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

fed_server = [95.87, 96.8, 97.07, 97.33, 97.4, 97.4, 97.33, 97.07, 97.47, 97.2, 97.2, 97.2, 97.33, 97.33, 97.4, 97.4, 97.4, 97.27, 97.27, 97.33, 97.4, 97.33, 97.4, 97.4, 97.47, 97.27, 97.4, 97.4, 97.33, 97.47, 97.53, 97.4, 97.67, 97.47, 97.4, 97.67, 97.47, 97.6, 97.67, 97.53, 97.53, 97.47, 97.53, 97.47, 97.67, 97.53, 97.67, 97.47, 97.6, 97.73]
fed_server_alpha_025 = [85.6, 83.67, 83.67, 84.2, 85.33, 85.93, 87.07, 87.53, 88.4, 88.87, 89.2, 89.67, 90.33, 90.6, 91.2, 91.73, 91.8, 91.4, 92.0, 92.13, 92.4, 92.47, 92.53, 92.4, 92.53, 92.73, 92.8, 92.47, 93.07, 93.2, 93.47, 93.6, 93.4, 93.07, 93.73, 93.73, 93.8, 93.8, 93.73, 93.8, 93.8, 93.93, 94.0, 94.07, 93.6, 93.93, 93.8, 93.87, 94.2, 94.27]
fed_server_alpha_050 = [93.53, 94.4, 94.93, 94.47, 94.87, 94.93, 95.2, 95.47, 95.8, 95.87, 95.53, 95.73, 96.13, 96.07, 96.4, 96.53, 96.33, 96.13, 96.4, 96.6, 96.33, 96.53, 96.53, 96.47, 96.53, 96.8, 96.93, 96.53, 96.47, 96.67, 96.73, 96.73, 96.67, 96.73, 96.73, 96.6, 96.73, 96.8, 96.73, 96.67, 96.53, 96.67, 97.0, 96.8, 96.93, 97.07, 96.73, 97.07, 97.0, 96.93]
fed_server_alpha_075 = [95.8, 96.33, 97.0, 97.0, 97.07, 97.33, 97.13, 97.07, 97.2, 97.53, 97.4, 97.2, 97.4, 97.4, 97.4, 97.4, 97.2, 97.27, 97.47, 97.27, 97.27, 97.2, 97.33, 97.4, 97.67, 97.4, 97.53, 97.27, 97.4, 97.4, 97.53, 97.47, 97.53, 97.6, 97.47, 97.53, 97.53, 97.67, 97.53, 97.53, 97.53, 97.47, 97.53, 97.73, 97.73, 97.6, 97.4, 97.73, 97.73, 97.47]
main_fed = [63.87, 67.2, 71.4, 74.13, 76.8, 78.8, 80.0, 80.73, 81.2, 81.87, 82.47, 82.6, 83.13, 83.47, 83.87, 84.4, 85.0, 85.07, 85.67, 85.8, 85.93, 86.07, 86.07, 86.4, 86.73, 86.53, 87.0, 87.07, 87.33, 87.2, 87.73, 87.73, 87.8, 87.73, 88.07, 87.87, 88.2, 88.27, 88.53, 88.6, 88.73, 88.73, 88.87, 89.07, 89.0, 89.13, 89.13, 89.27, 89.13, 89.0]
main_nn = [9.93, 96.4, 96.87, 97.07, 97.2, 97.2, 97.4, 97.6, 97.73, 97.53, 97.33, 97.6, 97.47, 97.47, 97.53, 97.6, 97.67, 97.6, 97.53, 97.47, 97.6, 97.53, 97.67, 97.67, 97.6, 97.53, 97.73, 97.53, 97.53, 97.53, 97.6, 97.53, 97.6, 97.6, 97.47, 97.53, 97.47, 97.6, 97.53, 97.53, 97.53, 97.6, 97.53, 97.6, 97.67, 97.53, 97.67, 97.6, 97.6, 97.6]

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
plt.ylim(70)
plt.legend(prop=legendFont)
plt.grid()
plt.show()
