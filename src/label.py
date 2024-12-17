import pickle

with open(r"data/interim/in_labelled.bin", "rb") as data_file:
    in_to_label = pickle.load(data_file)

# Ask the user how many entries to label
n = int(input("Set number of data to label: "))

# Counter to track labeled entries
labeled_count = 0

for index, row in in_to_label.iterrows():
    # Skip rows that already have a sentiment labeled
    if row["sentiment"] is not None:
        continue

    # Print the text of the current row
    print("\n\n\n\n\n\n------------------------------------------------------")
    print(f"Text to label ({index + 1}/{len(in_to_label)}): \n{row['text']}")

    # Ask the user to provide a label
    sentiment = input("\nProvide sentiment (1=positive, 2=neutral, 3=negative): ")
    print("------------------------------------------------------")

    # Assign the sentiment value
    in_to_label.at[index, "sentiment"] = int(sentiment)

    # Increment the labeled counter
    labeled_count += 1
    
    with open(r"data/interim/labelled.bin", "wb") as filehandler:
       pickle.dump(in_to_label, filehandler)

    # Break the loop if the required number of entries is labeled
    if labeled_count >= n:
        break

# Save the updated data back to the same file
with open(r"data/interim/in_labelled.bin", "wb") as filehandler:
    pickle.dump(in_to_label, filehandler)

print(f"\nLabeling complete. {labeled_count} entries labeled.")
