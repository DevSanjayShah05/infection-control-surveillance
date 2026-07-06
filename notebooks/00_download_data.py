"""
Downloads the real CMS datasets this project is built on. Queries the CMS
Provider Data Catalog metastore API for the current download URLs (CMS
periodically re-issues these with a new timestamped filename) rather than
hardcoding a URL that will eventually 404.
"""
import requests
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DATASETS = {
    "77hc-ibv8": "hai_hospital_raw.csv",             # Healthcare Associated Infections - Hospital
    "xubh-q36u": "hospital_general_info_raw.csv",    # Hospital General Information
}


def download(dataset_id: str, out_name: str):
    meta = requests.get(
        f"https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/{dataset_id}",
        timeout=30,
    ).json()
    download_url = meta["distribution"][0]["downloadURL"]
    print(f"Dataset: {meta['title']} (last modified {meta.get('modified')})")
    print(f"Downloading: {download_url}")

    resp = requests.get(download_url, timeout=120)
    resp.raise_for_status()
    out_path = DATA_DIR / out_name
    out_path.write_bytes(resp.content)
    print(f"Saved {len(resp.content) / 1e6:.1f} MB -> {out_path}\n")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id, out_name in DATASETS.items():
        download(dataset_id, out_name)


if __name__ == "__main__":
    main()
