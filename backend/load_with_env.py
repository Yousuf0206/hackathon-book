import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

# Now run the content loader
import sys
import argparse

# Add the backend src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.content_loader import load_content_to_qdrant, main as content_loader_main

def main():
    content_loader_main()

if __name__ == "__main__":
    main()