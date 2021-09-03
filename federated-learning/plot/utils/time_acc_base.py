# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def plot_time_acc(title, scale, xrange, fed_async, fed_avg, fed_sync, fed_localA, local_train, save_path=None,
                  is_acc=True):
    x = range(len(fed_async))
    x = [value * scale for value in x]

    fig, axes = plt.subplots()

    legend_font = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
    xy_label_font = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=19)
    title_font = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
    cs_xy_label_font = {'fontproperties': xy_label_font}
    cs_title_font = {'fontproperties': title_font}

    axes.plot(x, fed_async, label="BAFL", linewidth=3)
    axes.plot(x, fed_sync, label="BSFL", linestyle='--', alpha=0.5)
    axes.plot(x, fed_localA, label="APFL", linestyle='--', alpha=0.5)
    axes.plot(x, fed_avg, label="FedAVG", linestyle='--', alpha=0.5)
    axes.plot(x, local_train, label="Local Training", linestyle='--', alpha=0.5)

    axes.set_xlabel("Running Time (seconds)", **cs_xy_label_font)
    if is_acc:
        axes.set_ylabel("Average Test Accuracy (%)", **cs_xy_label_font)
    else:
        axes.set_ylabel("Mean Squared Error", **cs_xy_label_font)

    plt.title(title, **cs_title_font)
    plt.xticks(family='Times New Roman', fontsize=15)
    plt.yticks(family='Times New Roman', fontsize=15)
    plt.tight_layout()
    plt.xlim(0, xrange)
    plt.legend(prop=legend_font, loc='lower right')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_static_time_acc(title, scale, xrange, fed_async, fed_async_f05, fed_async_f10, fed_async_f15, save_path=None,
                         is_acc=True):
    x = range(len(fed_async))
    x = [value * scale for value in x]

    fig, axes = plt.subplots()

    legend_font = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
    xy_label_font = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=19)
    title_font = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=17)
    cs_xy_label_font = {'fontproperties': xy_label_font}
    cs_title_font = {'fontproperties': title_font}

    axes.plot(x, fed_async, label="BAFL", linewidth=3)
    axes.plot(x, fed_async_f05, label="f=0.5", linestyle='--', alpha=0.5)
    axes.plot(x, fed_async_f10, label="f=1.0", linestyle='--', alpha=0.5)
    axes.plot(x, fed_async_f15, label="f=1.5", linestyle='--', alpha=0.5)

    axes.set_xlabel("Running Time (seconds)", **cs_xy_label_font)
    if is_acc:
        axes.set_ylabel("Average Test Accuracy (%)", **cs_xy_label_font)
    else:
        axes.set_ylabel("Mean Squared Error", **cs_xy_label_font)

    plt.title(title, **cs_title_font)
    plt.xticks(family='Times New Roman', fontsize=15)
    plt.yticks(family='Times New Roman', fontsize=15)
    plt.tight_layout()
    plt.xlim(0, xrange)
    plt.legend(prop=legend_font, loc='lower right')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
