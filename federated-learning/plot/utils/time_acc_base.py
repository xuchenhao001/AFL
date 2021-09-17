# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt-get install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# input latex symbols in matplotlib
# https://stackoverflow.com/questions/43741928/matplotlib-raw-latex-epsilon-only-yields-varepsilon
plt.rcParams["mathtext.fontset"] = "cm"


# Plot size settings: "L", "M", "S"
# L: Single plot
# M: Three in a row
# S: Four in a row (bigger fonts)
def get_font_settings(size):
    if size == "L":
        font_size_dict = {"l": 17, "m": 15, "s": 13}
    elif size == "M":
        font_size_dict = {"l": 19, "m": 17, "s": 15}
    else:
        font_size_dict = {"l": 25, "m": 21, "s": 19}

    xy_label_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["l"])
    title_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["m"])
    legend_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["s"])
    ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', size=font_size_dict["s"])
    cs_xy_label_font = {'fontproperties': xy_label_font}
    cs_title_font = {'fontproperties': title_font}
    cs_xy_ticks_font = {'fontproperties': ticks_font}
    font_factory = {
        'legend_font': legend_font,
        'cs_xy_label_font': cs_xy_label_font,
        'cs_title_font': cs_title_font,
        'cs_xy_ticks_font': cs_xy_ticks_font,
    }
    return font_factory


def plot_time_acc(title, scale, xrange, fed_async, fed_avg, fed_sync, fed_localA, local_train, save_path=None,
                  is_acc=True, plot_size="L"):
    font_settings = get_font_settings(plot_size)
    x = range(len(fed_async))
    x = [value * scale for value in x]

    fig, axes = plt.subplots()

    axes.plot(x, fed_async, label="BAFL", linewidth=3)
    axes.plot(x, fed_sync, label="BSFL", linestyle='--', alpha=0.5)
    axes.plot(x, fed_localA, label="APFL", linestyle='--', alpha=0.5)
    axes.plot(x, fed_avg, label="FedAVG", linestyle='--', alpha=0.5)
    axes.plot(x, local_train, label="Local", linestyle='--', alpha=0.5)

    axes.set_xlabel("Running Time (seconds)", **font_settings.get("cs_xy_label_font"))
    if is_acc:
        axes.set_ylabel("Average Test Accuracy (%)", **font_settings.get("cs_xy_label_font"))
    else:
        axes.set_ylabel("Average Loss (MSE)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.tight_layout()
    plt.xlim(0, xrange)
    plt.legend(prop=font_settings.get("legend_font"), loc='lower right')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_static_time_acc(title, scale, xrange, fed_async, fed_async_f05, fed_async_f10, fed_async_f15, save_path=None,
                         is_acc=True, plot_size="L"):
    font_settings = get_font_settings(plot_size)
    x = range(len(fed_async))
    x = [value * scale for value in x]

    fig, axes = plt.subplots()

    axes.plot(x, fed_async, label="Dynamic", linewidth=3)
    axes.plot(x, fed_async_f05, label=r'$\mathbf{\epsilon}$ = 0.5', linestyle='--', alpha=0.5)
    axes.plot(x, fed_async_f10, label=r'$\epsilon$ = 1.0', linestyle='--', alpha=0.5)
    axes.plot(x, fed_async_f15, label=r'$\epsilon$ = 1.5', linestyle='--', alpha=0.5)

    axes.set_xlabel("Running Time (seconds)", **font_settings.get("cs_xy_label_font"))
    if is_acc:
        axes.set_ylabel("Average Test Accuracy (%)", **font_settings.get("cs_xy_label_font"))
    else:
        axes.set_ylabel("Average Loss (MSE)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.tight_layout()
    plt.xlim(0, xrange)
    plt.legend(prop=font_settings.get("legend_font"), loc='lower right')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_time_cost(title, yrange, fed_async, fed_avg, fed_sync, fed_localA, local_train, save_path=None, plot_size="L"):
    font_settings = get_font_settings(plot_size)
    x = range(1, len(fed_async)+1)

    fig, axes = plt.subplots()

    axes.plot(x, fed_async, label="BAFL", linewidth=3)
    axes.plot(x, fed_sync, label="BSFL", linestyle='--', alpha=0.5)
    axes.plot(x, fed_localA, label="APFL", linestyle='--', alpha=0.5)
    axes.plot(x, fed_avg, label="FedAVG", linestyle='--', alpha=0.5)
    axes.plot(x, local_train, label="Local", linestyle='--', alpha=0.5)

    axes.set_xlabel("Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Average Time (s)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.tight_layout()
    plt.ylim(0, yrange)
    plt.legend(prop=font_settings.get("legend_font"), loc='upper right')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_time_historgram(title, fed_async, fed_avg, fed_sync, fed_localA, local_train, save_path=None, plot_size="L"):
    font_settings = get_font_settings(plot_size)
    x = ["Train", "Test", "Communication", "Round"]

    fig, axes = plt.subplots()

    width = 0.15  # the width of the bars
    axes.bar([(p - width * 2) for p in range(len(x))], height=fed_async, width=width, label="BAFL", hatch='x')
    axes.bar([p - width for p in range(len(x))], height=fed_sync, width=width, label="BSFL", hatch='o')
    axes.bar(range(len(x)), height=fed_localA, width=width, label="APFL", hatch='+')
    axes.bar([p + width for p in range(len(x))], height=fed_avg, width=width, label="FedAVG", hatch='*')
    axes.bar([(p + width * 2) for p in range(len(x))], height=local_train, width=width, label="Local",
             hatch='/')

    plt.xticks(range(len(x)), x)
    axes.set_xlabel("Steps in the Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Average Time (s)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.tight_layout()
    plt.legend(prop=font_settings.get("legend_font"), loc='upper left')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_poisoning_time_acc(title, scale, xrange, acc_average, acc_node_list, save_path=None, is_acc=True, plot_size="L"):
    font_settings = get_font_settings(plot_size)
    x = range(len(acc_average))
    x = [value * scale for value in x]

    fig, axes = plt.subplots()

    axes.plot(x, acc_average, label="Average", linewidth=3)
    axes.plot(x, acc_node_list[0], label="Node 1", linestyle='--', alpha=0.5)
    axes.plot(x, acc_node_list[1], label="Node 2", linestyle='--', alpha=0.5)
    axes.plot(x, acc_node_list[2], label="Node 3", linestyle='--', alpha=0.5)
    axes.plot(x, acc_node_list[3], label="Node 4", linestyle='--', alpha=0.5)
    axes.plot(x, acc_node_list[4], label="Node 5", linestyle='--', alpha=0.5)

    axes.set_xlabel("Running Time (seconds)", **font_settings.get("cs_xy_label_font"))
    if is_acc:
        axes.set_ylabel("Test Accuracy (%)", **font_settings.get("cs_xy_label_font"))
    else:
        axes.set_ylabel("Average Loss (MSE)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.tight_layout()
    plt.xlim(0, xrange)
    plt.legend(prop=font_settings.get("legend_font"), loc='lower right')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_ddos_acc(title, scale, xrange, ddos_00, ddos_80, ddos_90, save_path=None, is_acc=True, plot_size="L"):
    font_settings = get_font_settings(plot_size)
    x = range(len(ddos_00))
    x = [value * scale for value in x]

    fig, axes = plt.subplots()

    axes.plot(x, ddos_00, label="Normal")
    axes.plot(x, ddos_80, label="DDoS 80%")
    axes.plot(x, ddos_90, label="DDoS 90%")

    axes.set_xlabel("Running Time (seconds)", **font_settings.get("cs_xy_label_font"))
    if is_acc:
        axes.set_ylabel("Average Test Accuracy (%)", **font_settings.get("cs_xy_label_font"))
    else:
        axes.set_ylabel("Average Loss (MSE)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.tight_layout()
    plt.xlim(0, xrange)
    plt.legend(prop=font_settings.get("legend_font"), loc='lower right')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

