import json
import pickle

from config import client, cookie_json_str, get_run_input, urls

actor_runs_num = len(urls) # type: ignore
cookies = []
inputs = []
runs = []
pages = []

cookie_parsed = json.loads(cookie_json_str) # type: ignore
cookies.append(cookie_parsed)

for x in range(actor_runs_num):
    inputs.append(get_run_input(urls[x], cookies[0])) # type: ignore
    
# Scrape the data 
for x in range(actor_runs_num):
    run = client.actor("kfiWbq3boy3dWKbiL").call(run_input=inputs[x])
    pages.append(client.dataset(run["defaultDatasetId"]).list_items(clean=True)) # type: ignore

# Save the data
for x in range(actor_runs_num):
    with open(rf"data/raw/data_raw{x+1}.bin", "wb") as filehandler:
        pickle.dump(pages[x].items, filehandler)
        