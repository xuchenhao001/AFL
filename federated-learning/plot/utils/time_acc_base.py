# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# Font settings for plots start #

# Three in a row #
font_size_dict = {"h": 19, "m": 17, "l": 15}

# Four in a row (bigger fonts) #
# font_size_dict = {"h": 25, "m": 21, "l": 19}

# Font settings for plots end #

xy_label_font = font_manager.FontProperties(
    family='Times New Roman', weight='bold', style='normal', size=font_size_dict["h"])
title_font = font_manager.FontProperties(
    family='Times New Roman', weight='bold', style='normal', size=font_size_dict["m"])
legend_font = font_manager.FontProperties(
    family='Times New Roman', weight='bold', style='normal', size=font_size_dict["l"])
ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', size=font_size_dict["l"])
cs_xy_label_font = {'fontproperties': xy_label_font}
cs_title_font = {'fontproperties': title_font}
cs_xy_ticks_font = {'fontproperties': ticks_font}


def plot_time_acc(title, scale, xrange, fed_async, fed_avg, fed_sync, fed_localA, local_train, save_path=None,
                  is_acc=True):
    x = range(len(fed_async))
    x = [value * scale for value in x]

    fig, axes = plt.subplots()

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
    plt.xticks(**cs_xy_ticks_font)
    plt.yticks(**cs_xy_ticks_font)
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
    plt.xticks(**cs_xy_ticks_font)
    plt.yticks(**cs_xy_ticks_font)
    plt.tight_layout()
    plt.xlim(0, xrange)
    plt.legend(prop=legend_font, loc='lower right')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_time_cost(title, yrange, fed_async, fed_avg, fed_sync, fed_localA, local_train, save_path=None):
    x = range(1, len(fed_async)+1)

    fig, axes = plt.subplots()

    axes.plot(x, fed_async, label="BAFL", linewidth=3)
    axes.plot(x, fed_sync, label="BSFL", linestyle='--', alpha=0.5)
    axes.plot(x, fed_localA, label="APFL", linestyle='--', alpha=0.5)
    axes.plot(x, fed_avg, label="FedAVG", linestyle='--', alpha=0.5)
    axes.plot(x, local_train, label="Local Training", linestyle='--', alpha=0.5)

    axes.set_xlabel("Round Number", **cs_xy_label_font)
    axes.set_ylabel("Average Time (s)", **cs_xy_label_font)

    plt.title(title, **cs_title_font)
    plt.xticks(**cs_xy_ticks_font)
    plt.yticks(**cs_xy_ticks_font)
    plt.tight_layout()
    plt.ylim(0, yrange)
    plt.legend(prop=legend_font, loc='upper right')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
