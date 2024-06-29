import os
import subprocess
import logging
from typing import List, Tuple
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CVCDownloader(ABC):
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

class CVC1Downloader(CVCDownloader):
    """Downloader for Common Voice Corpus 1 dataset."""
    def __init__(self, output_dir: str):
        super().__init__("CVC1", output_dir)
        self.tar_file = "common_voice_corpus1.tar.tar"
        self.url = "https://zenodo.org/records/12588635/files/common_voice_corpus1.tar.tar?download=1"

    def download(self) -> None:
        subprocess.run(["wget", "-O", os.path.join("data", self.tar_file), self.url], check=True)

    def extract(self) -> None:
        subprocess.run(["tar", "-xvf", os.path.join("data", self.tar_file), "-C", self.output_dir], check=True)

    def cleanup(self) -> None:
        os.remove(os.path.join("data", self.tar_file))

class CVCDatasetProcessor:
    """Processor for Common Voice Corpus datasets."""
    def __init__(self, datasets: List[Tuple[str, str]]):
        self.datasets = datasets

    def process_datasets(self) -> None:
        """Process all specified datasets."""
        for dataset_name, output_dir in self.datasets:
            if dataset_name == "CVC1":
                downloader = CVC1Downloader(output_dir)
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
        ("CVC1", "cvc_1"),
    ]
    processor = CVCDatasetProcessor(datasets)
    processor.process_datasets()

if __name__ == "__main__":
    main()
