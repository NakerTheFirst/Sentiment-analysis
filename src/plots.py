import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

with open(r"data/interim/in_labelled.bin", "rb") as data_file:
    in_labelled = pickle.load(data_file)

# Count the number of positive, neutral, and negative sentiments
positive_count = len(in_labelled.loc[in_labelled["sentiment"] == 1])
neutral_count = len(in_labelled.loc[in_labelled["sentiment"] == 2])
negative_count = len(in_labelled.loc[in_labelled["sentiment"] == 3])

# Calculate the percentage of each sentiment
pos_per = round(positive_count / 500 * 100, 2)
neu_per = round(neutral_count / 500 * 100, 2)
neg_per = round(negative_count / 500 * 100, 2)

# Print the sentiment distribution
print(f"Positive:\t {positive_count},\t {pos_per}%")
print(f"Neutral:\t {neutral_count},\t {neu_per}%")
print(f"Negative:\t {negative_count},\t {neg_per}%")

map_sentiment_int_to_string = {
    1: "Positive",
    2: "Neutral",
    3: "Negative"
}

#* Plot the internal data sentiment distribution
in_labelled["sentiment"] = in_labelled["sentiment"].map(map_sentiment_int_to_string)
hist_plot = sns.histplot(in_labelled["sentiment"], discrete=True, stat="percent")
fig = hist_plot.get_figure()
plt.title('Sentiment distribution')
plt.ylabel('Percent')
plt.xlabel('Sentiment')
plt.show()

#* Plot TL data sentiment distribution
# Placeholder

# Save the plot
# fig.savefig("reports/figures/sentiment_hist.png") # type: ignore
plt.close()

#* Generate word clouds of data
# Combine all text into one string
text = " ".join(in_labelled["text"])

# Create and generate the wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Create a new figure with a specific size
plt.figure(figsize=(10, 5))

# Display the wordcloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word cloud of LiknedIn data')
plt.axis('off')

# Save the figure before showing it
# plt.savefig("reports/figures/wordcloud.png", bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# Close the figure
plt.close()
