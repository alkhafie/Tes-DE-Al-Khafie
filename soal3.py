import os
import json
import asyncio
import httpx
from bs4 import BeautifulSoup
from polars import DataFrame
from tqdm import tqdm

BASE_URL = "https://www.fortiguard.com/encyclopedia?type=ips&risk={}&page={}"
MAX_PAGES = [10, 15, 20, 25, 30]
OUTPUT_DIR = "datasets"


async def fetch_page(session, level, page):
    url = BASE_URL.format(level, page)
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()
    except httpx.HTTPError as e:
        print(f"HTTP error {e.response.status_code} while fetching {url}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None


def parse_page(html):
    soup = BeautifulSoup(html, "html.parser")
    articles = []
    for article in soup.find_all("div", class_="content-item"):
        title = article.find("h4").text.strip()
        link = article.find("a")["href"].strip()
        articles.append({"title": title, "link": link})
    return articles


async def scrape_level(session, level, max_pages):
    skipped_pages = []
    data = []
    for i in tqdm(range(1, max_pages + 1), desc=f"Scraping Level {level}"):
        html = await fetch_page(session, level, i)
        if html is None:
            skipped_pages.append(i)
            continue

        articles = parse_page(html)
        data.extend(articles)

    return data, skipped_pages


async def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    tasks = []
    async with httpx.AsyncClient() as session:
        for level, max_pages in enumerate(MAX_PAGES, start=1):
            tasks.append(scrape_level(session, level, max_pages))

        results = await asyncio.gather(*tasks)

    for level, (level_data, skipped_pages) in enumerate(results, start=1):
        # Save CSV file
        df = DataFrame(level_data)
        csv_path = os.path.join(OUTPUT_DIR, f"forti_lists_{level}.csv")
        df.write_csv(csv_path)

        # Save skipped pages to JSON
        skipped_data = {"level": level, "skipped_pages": skipped_pages}
        json_path = os.path.join(OUTPUT_DIR, "skipped.json")
        with open(json_path, "w") as json_file:
            json.dump(skipped_data, json_file)

if __name__ == "__main__":
    asyncio.run(main())
