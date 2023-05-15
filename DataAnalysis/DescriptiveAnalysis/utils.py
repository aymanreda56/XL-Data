import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

def plots(x, y, xlabel, ylabel, title, style='dark_background', y_limit=False,\
           line_plot=False, stacked=False, y2=None, label2=None,horizontal=False):
    plt.figure(figsize=(20, 10))
    plt.style.use(style)
    plt.grid(False)

    # Define the color palette
    colors = sns.color_palette('Blues', len(x))[::-1]

    if line_plot:
        plt.plot(x, y, color=colors)

    elif stacked:
        plt.bar(x, y, color=colors[len(colors) // 2])
        plt.bar(x, y2, bottom=y, label=label2, color=colors[-1])
        plt.legend()

    elif horizontal:
        colors = colors[::-1]
        plt.barh(x, y, color=colors)
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)

    else:
        plt.bar(x, y, color=colors)

    if not horizontal:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90)

    if y_limit:
        plt.ylim(ymin=0)

    plt.show()


def pie_plots(x_perc, labels, title, style='dark_background',percentage=True):
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.style.use(style)

    # Normalize the values in x to add up to 100

    if percentage:
        total = sum(x_perc)
        x_perc = [val/total*100 for val in x_perc]

    # Define the color palette
    colors = sns.color_palette('Blues', len(x_perc))[::-1]

    plt.pie(x_perc, labels=[f"{label}: {value:.1f}%" for label, value in zip(labels, x_perc)],
            autopct='', labeldistance=1.1, textprops={'fontsize': 14}, colors=colors)
    plt.title(title, fontsize=16)
    plt.show()


def scatter_plot_legend(x, y, hue, xlabel, ylabel, color_label, title, style='dark_background'):

    # Create a dictionary that maps each unique content to a unique color
    unique_content = list(set(hue))
    color_map = dict(zip(unique_content, np.linspace(0, 1, len(unique_content))))

    # Map the content to a sequence of numbers based on the color map
    colors = [color_map[content] for content in hue]

    # Create a custom legend using the color map
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                              markerfacecolor=plt.cm.rainbow(value),
                              markersize=10) for key, value in color_map.items()]
    
    plt.figure(figsize=(20, 10))
    plt.style.use(style)
    plt.grid(False)
    plt.scatter(x, y, c=colors, cmap='rainbow')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(handles=legend_elements, loc='best')

    plt.title(title)
    plt.show()
