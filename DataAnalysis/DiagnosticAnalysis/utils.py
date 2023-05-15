import numpy as np
from matplotlib import pyplot as plt

def sub_blot(df, cat_col, grouby_col, col, title="",plot_kind='bar', style='dark_background', figsize=(30, 20)):
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    plt.style.use(style)
    for i, (genre, ax) in enumerate(zip(df[cat_col].unique(), axes.flatten())):
        df_genre = df[df[cat_col] == genre]
        publisher_sales = df_genre.groupby(grouby_col)[col].mean().sort_values(ascending=False).head(10)
        publisher_sales.plot(kind=plot_kind, ax=ax, title=genre)
    plt.tight_layout()
    plt.show()
