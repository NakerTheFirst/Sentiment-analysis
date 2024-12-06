from os import getenv

from apify_client import ApifyClient
from dotenv import load_dotenv

# Load the dotenv file
if not load_dotenv():
    raise Exception("Error: Environment variables not set")

# Initialise the ApifyClient with API token
API_KEY = getenv("API_KEY")
urls = getenv("URLS")
cookie_json_str = getenv("COOKIE")
client = ApifyClient(API_KEY)

def get_run_input(url: list[str], cookie: str) -> dict:
    """Return run_inp dictionary based on given cookie and url

    Args:
        cookie (str): JSON to Python dict parsed cookie
        url (str): URL to be scraped

    Returns:
        dict: Input run settings
    """
    
    run_inp = {
        "cookie": cookie,
        "urls": url,
        "deepScrape": True,
        "rawData": False,
        "minDelay": 2,
        "maxDelay": 4,
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyCountry": "US",
        }
    }

    return run_inp
