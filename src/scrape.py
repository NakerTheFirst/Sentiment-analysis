import pickle

from config import client, run_input

# Start scraping and wait for it to finish
run = client.actor("kfiWbq3boy3dWKbiL").call(run_input=run_input, timeout_secs=10)

# Get and save raw data to file via pickle
list_page = client.dataset(run["defaultDatasetId"]).list_items(clean=True)

with open(r"data/raw/data_raw.bin", "wb") as filehandler:
    pickle.dump(list_page.items, filehandler)
