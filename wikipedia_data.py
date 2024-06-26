import os
import sys
import requests
import bz2
import mwparserfromhell

def download_wiki_dump(url, output_file):
    print(f"Downloading Wikipedia dump from {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    downloaded = 0

    with open(output_file, 'wb') as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            downloaded += size
            progress = int(50 * downloaded / total_size)
            sys.stdout.write(f"\r[{'=' * progress}{' ' * (50-progress)}] {downloaded}/{total_size} bytes")
            sys.stdout.flush()
    print("\nDownload completed.")

def extract_wiki_data(input_file, output_file):
    print("Extracting Wikipedia data...")
    with bz2.open(input_file, 'rt', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():
                wikicode = mwparserfromhell.parse(line)
                text = wikicode.strip_code()
                outfile.write(text + '\n')
    print("Extraction completed.")

def clean_extracted_data(input_file, output_file):
    print("Cleaning extracted data...")
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():
                clean_line = ' '.join(line.split())
                outfile.write(clean_line + '\n')
    print("Cleaning completed.")

def main():
    wiki_dump_url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2"
    dump_file = "./data/wiki_dump.xml.bz2"
    extracted_file = "./data/extracted_wiki_data.txt"
    final_output_file = "./data/training_wiki_data.txt"

    download_wiki_dump(wiki_dump_url, dump_file)
    extract_wiki_data(dump_file, extracted_file)
    clean_extracted_data(extracted_file, final_output_file)

    os.remove(dump_file)
    os.remove(extracted_file)

    print(f"Wikipedia data has been successfully extracted and saved to {final_output_file}")

if __name__ == "__main__":
    main()
