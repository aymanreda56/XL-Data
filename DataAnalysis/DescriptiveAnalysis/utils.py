import matplotlib.pyplot as plt
import numpy as np

def plots(x,y,xlabel,ylabel,title, style='dark_background',y_limit=False, line_plot=False, stacked=False, y2=None, label2=None):
    plt.figure(figsize=(20, 10))
    plt.style.use(style)
    plt.grid(False)

    if line_plot:
        plt.plot(x,y)
    else:
        plt.bar(x,y)

        if stacked:
            plt.bar(x, y2, bottom=y, label=label2)
            plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90)

    if y_limit:
        plt.ylim(0, 5)
    
    plt.show()


def pie_plots(x,labels,title, style='dark_background'):
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.style.use(style)
    plt.pie(x, labels=labels, autopct='%1.1f%%', labeldistance=1.1, textprops={'fontsize': 14})
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
