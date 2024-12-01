import json
import pickle

from config import client, cookies_json_str, get_run_input, urls

actor_runs_num = len(urls)
cookies = []
inputs = []
runs = []
pages = []

for unparsed_cookie in cookies_json_str:
    cookie_parsed = json.loads(unparsed_cookie)
    cookies.append(cookie_parsed)

for x in range(actor_runs_num):
    inputs.append(get_run_input(urls[x], cookies[x]))
    
# Scrape the data 
for x in range(actor_runs_num):
    run = client.actor("kfiWbq3boy3dWKbiL").call(run_input=inputs[x], timeout_secs=15)
    pages.append(client.dataset(run["defaultDatasetId"]).list_items(clean=True)) # type: ignore

# Save the data
for x in range(actor_runs_num):
    with open(rf"data/raw/data_raw{x+1}.bin", "wb") as filehandler:
        pickle.dump(pages[x].items, filehandler)
        