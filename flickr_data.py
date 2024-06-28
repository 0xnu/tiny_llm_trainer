import os
import subprocess
import logging
from typing import List, Tuple
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FlickrDownloader(ABC):
    """Abstract base class for dataset downloaders."""
    def __init__(self, dataset_name: str, output_dir: str):
        self.dataset_name = dataset_name
        self.output_dir = os.path.join("data", output_dir)

    @abstractmethod
    def download(self) -> None:
        """Download the dataset."""
        pass

    @abstractmethod
    def extract(self) -> None:
        """Extract the downloaded dataset."""
        pass

    def cleanup(self) -> None:
        """Clean up temporary files."""
        pass

    def process(self) -> None:
        """Process the dataset: download, extract, and cleanup."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.download()
        self.extract()
        self.cleanup()
        logging.info(f"Downloaded {self.dataset_name} dataset successfully to {self.output_dir}")

class Flickr8kDownloader(FlickrDownloader):
    """Downloader for Flickr8k dataset."""
    def __init__(self, output_dir: str):
        super().__init__("Flickr8k", output_dir)
        self.zip_file = "flickr8k.zip"
        self.url = "https://zenodo.org/records/12572919/files/flickr8k.zip?download=1"

    def download(self) -> None:
        subprocess.run(["wget", "-O", os.path.join("data", self.zip_file), self.url], check=True)

    def extract(self) -> None:
        subprocess.run(["unzip", "-q", os.path.join("data", self.zip_file), "-d", self.output_dir], check=True)

    def cleanup(self) -> None:
        os.remove(os.path.join("data", self.zip_file))

class Flickr30kDownloader(FlickrDownloader):
    """Downloader for Flickr30k dataset."""
    def __init__(self, output_dir: str):
        super().__init__("Flickr30k", output_dir)
        self.part_files = [f"flickr30k_part0{i}" for i in range(3)]
        self.urls = [
            "https://zenodo.org/records/12572919/files/flickr30k_part00?download=1",
            "https://zenodo.org/records/12572919/files/flickr30k_part01?download=1",
            "https://zenodo.org/records/12572919/files/flickr30k_part02?download=1"
        ]

    def download(self) -> None:
        for part, url in zip(self.part_files, self.urls):
            subprocess.run(["wget", "-O", os.path.join("data", part), url], check=True)

    def extract(self) -> None:
        # Combine parts
        with open(os.path.join(self.output_dir, "flickr30k.zip"), 'wb') as outfile:
            for part in self.part_files:
                with open(os.path.join("data", part), 'rb') as infile:
                    outfile.write(infile.read())
        
        subprocess.run(["unzip", "-q", os.path.join(self.output_dir, "flickr30k.zip"), "-d", self.output_dir], check=True)

    def cleanup(self) -> None:
        os.remove(os.path.join(self.output_dir, "flickr30k.zip"))
        for part in self.part_files:
            os.remove(os.path.join("data", part))

class FlickrDatasetProcessor:
    """Processor for Flickr datasets."""
    def __init__(self, datasets: List[Tuple[str, str]]):
        self.datasets = datasets

    def process_datasets(self) -> None:
        """Process all specified datasets."""
        for dataset_name, output_dir in self.datasets:
            if dataset_name == "Flickr8k":
                downloader = Flickr8kDownloader(output_dir)
            elif dataset_name == "Flickr30k":
                downloader = Flickr30kDownloader(output_dir)
            else:
                logging.warning(f"Unknown dataset: {dataset_name}")
                continue

            try:
                downloader.process()
            except subprocess.CalledProcessError as e:
                logging.error(f"Error processing {dataset_name}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error processing {dataset_name}: {e}")

def main() -> None:
    datasets = [
        ("Flickr8k", "flickr8k"),
        ("Flickr30k", "flickr30k"),
    ]
    processor = FlickrDatasetProcessor(datasets)
    processor.process_datasets()

if __name__ == "__main__":
    main()