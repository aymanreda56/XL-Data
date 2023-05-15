import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plots(x, y, xlabel, ylabel, title, style='dark_background', y_limit=False, line_plot=False, stacked=False, y2=None, label2=None):
    plt.figure(figsize=(20, 10))
    plt.style.use(style)
    plt.grid(False)

    # Define the color palette
    colors = sns.color_palette('Blues', len(x))[::-1]

    if line_plot:
        plt.plot(x, y, color=colors[0])
    else:
        plt.bar(x, y, color=colors)

        if stacked:
            plt.bar(x, y2, bottom=y, label=label2, color=colors[1])
            plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90)

    # if y_limit:
    #     plt.ylim(0, 5)

    # always start y axis at 0
    plt.ylim(ymin=0)

    plt.show()

def pie_plots(x, labels, title, style='dark_background'):
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.style.use(style)

    # Define the color palette
    colors = sns.color_palette('Blues', len(x))[::-1]

    plt.pie(x, labels=[f"{label}: {value:.1f}%" for label, value in zip(labels, x)],
            autopct='', labeldistance=1.1, textprops={'fontsize': 14}, colors=colors)
    plt.title(title, fontsize=16)
    plt.show()


def scatter_plot_colorbar(x,y,hue,xlabel,ylabel,color_label,title, style='dark_background'):

    # Create a dictionary that maps each unique content to a unique color
    unique_content = list(set(hue))
    color_map = dict(zip(unique_content, np.linspace(0, 1, len(unique_content))))

    # Map the content  to a sequence of numbers based on the color map
    colors = [color_map[content] for content in hue]

    plt.style.use(style)
    plt.grid(False)
    plt.scatter(x, y, c=colors, cmap='rainbow')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    cb = plt.colorbar()
    cb.set_ticks(list(color_map.values()))
    cb.set_ticklabels(list(color_map.keys()))
    cb.set_label(color_label)

    plt.title(title)
    plt.show()
