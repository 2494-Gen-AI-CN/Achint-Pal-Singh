
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import tempfile
import shutil

DATA_DIR = Path(__file__).resolve().parent / "data"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/232/"
    "restaurant+consumer+data.zip"
)

FILES = [
    "rating_final.csv",
    "geoplaces2.csv",
    "chefmozcuisine.csv",
    "chefmozaccepts.csv",
    "chefmozparking.csv",
    "chefmozhours4.csv",
    "usercuisine.csv",
    "userpayment.csv",
    "userprofile.csv",
]


def download_dataset():
    """Download the nine UCI CSVs used for ratings, restaurant details, and saved preferences."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading restaurant dataset from UCI...")

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        zip_path = temp_dir / "restaurants.zip"

        urlretrieve(DATASET_URL, zip_path)

        with ZipFile(zip_path) as archive:
            archive.extractall(temp_dir / "extracted")

        extracted = temp_dir / "extracted"

        for filename in FILES:
            matches = list(extracted.rglob(filename))

            if not matches:
                raise FileNotFoundError(
                    f"{filename} missing from downloaded dataset"
                )

            destination = DATA_DIR / filename
            shutil.copy2(matches[0], destination)

            print(f"Saved: {destination}")

    print("\nAll dataset files downloaded successfully!")


if __name__ == "__main__":
    download_dataset()
