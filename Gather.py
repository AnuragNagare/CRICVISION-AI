import requests
import json
import csv
import os
from datetime import datetime
import time

# Create data directory if it doesn't exist
DATA_DIR = r"D:\Git hub project\New folder\Data"
os.makedirs(DATA_DIR, exist_ok=True)

class CricketDataScraper:
    def __init__(self):
        self.data_dir = DATA_DIR
        
    def download_cricsheet_data(self):
        """
        Download Cricsheet data (JSON format)
        Cricsheet provides free ball-by-ball data
        """
        print("Downloading Cricsheet data...")
        
        # Cricsheet GitHub repository URLs for different formats
        base_url = "https://cricsheet.org/downloads/"
        
        datasets = {
            "t20s_male_json": "t20s_male_json.zip",
            "odis_male_json": "odis_male_json.zip",
            "tests_male_json": "tests_male_json.zip",
            "ipl_json": "ipl_json.zip",
            "bbl_json": "bbl_json.zip"
        }
        
        for name, filename in datasets.items():
            try:
                print(f"Downloading {name}...")
                url = base_url + filename
                response = requests.get(url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    filepath = os.path.join(self.data_dir, filename)
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"✓ Downloaded {name} to {filepath}")
                else:
                    print(f"✗ Failed to download {name}: Status {response.status_code}")
                    
                time.sleep(1)  # Be respectful with requests
                
            except Exception as e:
                print(f"✗ Error downloading {name}: {str(e)}")
    
    def fetch_cricapi_data(self, api_key):
        """
        Fetch data from CricAPI (requires free API key)
        Get your key from: https://www.cricapi.com/
        """
        print("\nFetching CricAPI data...")
        
        base_url = "https://api.cricapi.com/v1/"
        
        endpoints = {
            "current_matches": "currentMatches",
            "match_info": "matches",
            "series": "series"
        }
        
        for name, endpoint in endpoints.items():
            try:
                print(f"Fetching {name}...")
                url = f"{base_url}{endpoint}?apikey={api_key}&offset=0"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    filepath = os.path.join(self.data_dir, f"cricapi_{name}_{datetime.now().strftime('%Y%m%d')}.json")
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    print(f"✓ Saved {name} to {filepath}")
                else:
                    print(f"✗ Failed to fetch {name}: Status {response.status_code}")
                    
                time.sleep(1)
                
            except Exception as e:
                print(f"✗ Error fetching {name}: {str(e)}")
    
    def scrape_espncricinfo_recent_matches(self):
        """
        Scrape recent match IDs from ESPNcricinfo
        Note: For detailed data, you'd need to parse individual match pages
        """
        print("\nScraping ESPNcricinfo recent matches...")
        
        try:
            # This is a simplified example - real scraping would need more robust parsing
            url = "https://www.espncricinfo.com/live-cricket-score"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                filepath = os.path.join(self.data_dir, f"espncricinfo_recent_{datetime.now().strftime('%Y%m%d')}.html")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"✓ Saved ESPNcricinfo page to {filepath}")
                print("Note: You'll need BeautifulSoup to parse HTML for structured data")
            else:
                print(f"✗ Failed to fetch ESPNcricinfo: Status {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error scraping ESPNcricinfo: {str(e)}")
    
    def process_cricsheet_json(self, json_file):
        """
        Process a Cricsheet JSON file and extract bowling statistics
        """
        print(f"\nProcessing {json_file}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
            
            bowling_stats = []
            
            # Extract innings data
            innings = match_data.get('innings', [])
            
            for inning in innings:
                overs = inning.get('overs', [])
                
                for over in overs:
                    bowler = over.get('bowler', 'Unknown')
                    
                    for delivery in over.get('deliveries', []):
                        ball_data = {
                            'match_id': match_data.get('info', {}).get('match_id', 'N/A'),
                            'bowler': bowler,
                            'runs': delivery.get('runs', {}).get('total', 0),
                            'wicket': 1 if 'wickets' in delivery else 0,
                            'extras': delivery.get('runs', {}).get('extras', 0)
                        }
                        bowling_stats.append(ball_data)
            
            # Save processed data
            output_file = os.path.join(self.data_dir, f"processed_{os.path.basename(json_file).replace('.json', '.csv')}")
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if bowling_stats:
                    writer = csv.DictWriter(f, fieldnames=bowling_stats[0].keys())
                    writer.writeheader()
                    writer.writerows(bowling_stats)
                    print(f"✓ Processed data saved to {output_file}")
                    
        except Exception as e:
            print(f"✗ Error processing file: {str(e)}")

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("Cricket Data Scraper")
    print("=" * 60)
    
    scraper = CricketDataScraper()
    
    # Your CricAPI key
    CRICAPI_KEY = "da998a01-9b16-49af-8c6c-d8a26aef1181"
    
    print("\nOptions:")
    print("1. Download Cricsheet data (Free, no API key needed)")
    print("2. Fetch CricAPI data (Using your API key)")
    print("3. Scrape ESPNcricinfo (Basic scraping)")
    print("4. All of the above")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice in ['1', '4']:
        scraper.download_cricsheet_data()
    
    if choice in ['2', '4']:
        scraper.fetch_cricapi_data(CRICAPI_KEY)
    
    if choice in ['3', '4']:
        scraper.scrape_espncricinfo_recent_matches()
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print(f"Check your data directory: {DATA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()