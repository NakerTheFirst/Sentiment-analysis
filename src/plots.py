from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


def create_hist(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    xlabel: str,
    column: str = "label",
    show: bool = False,
    save: bool = False,
    save_path: Optional[str] = None
) -> None:
    """
    Create a histogram plot using seaborn.
    
    Args:
        df: DataFrame containing the data to plot
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label
        column: Column name to plot (defaults to 'sentiment')
        show: Whether to display the plot
        save: Whether to save the plot
        save_path: Path where to save the plot if save is True
    """
    # Create the histogram
    hist_plot = sns.histplot(df[column], discrete=True, stat="percent")
    fig = hist_plot.get_figure()
    
    # Set the labels and title
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    
    # Save if requested
    if save and save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show if requested
    if show:
        plt.show()
    
    plt.close()

def create_word_cloud(
    df: pd.DataFrame,
    title: str,
    text_column: str = "text",
    width: int = 800,
    height: int = 400,
    show: bool = False,
    save: bool = False,
    save_path: Optional[str] = None
) -> None:
    """
    Create and display a word cloud from DataFrame text.
    
    Args:
        df: DataFrame containing the text data
        title: Plot title
        text_column: Column name containing text data
        width: Width of the word cloud
        height: Height of the word cloud
        show: Whether to display the plot
        save: Whether to save the plot
        save_path: Path where to save the plot if save is True
    """
    # Combine all text into one string
    text = " ".join(df[text_column])
    
    # Create and generate the wordcloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white'
    ).generate(text)
    
    # Create figure with specific size
    plt.figure(figsize=(10, 5))
    
    # Display the wordcloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    
    # Save if requested
    if save and save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show if requested
    if show:
        plt.show()
    
    plt.close()

data_eval = pd.read_csv("data/processed/data_eval.csv")
data_tl = pd.read_csv("data/processed/data_tl.csv")

# Map sentiment values to strings if needed
map_sentiment_int_to_string = {
    0: "Positive",
    1: "Neutral",
    2: "Negative"
}

data_eval["label"] = data_eval["label"].map(map_sentiment_int_to_string)
data_tl["label"] = data_tl["label"].map(map_sentiment_int_to_string)

#! TODO: Convert figures to Polish...
# Create internal data sentiment histogram
create_hist(
    df=data_eval,
    title='Evaluation data sentiment distribution',
    ylabel='Percent',
    xlabel='Sentiment',
    show=True,
    save=False,
    save_path="reports/figures/sentiment_hist.png"
)

# Create transfer learning data sentiment histogram
create_hist(
    df=data_tl,
    title='Transfer learning dataset sentiment distribution',
    ylabel='Percent',
    xlabel='Sentiment',
    show=True,
    save=False,
    save_path="reports/figures/sentiment_hist_tl.png"
)

# Create internal data word cloud
create_word_cloud(
    df=data_eval,
    title='Word cloud of evaluation data',
    show=False,
    save=False,
    save_path="reports/figures/wordcloud.png"
)

# Create transfer learning data word cloud
create_word_cloud(
    df=data_tl,
    title='Word cloud of transfer learning data',
    show=False,
    save=False,
    save_path="reports/figures/wordcloud_tl.png"
)
