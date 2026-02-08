import io
from pathlib import Path

import pdfplumber
import pandas as pd
from aidial_client import Dial
from bs4 import BeautifulSoup


class DialFileContentExtractor:

    def __init__(self, endpoint: str, api_key: str):
        self.client = Dial(base_url=endpoint, api_key=api_key)

    def extract_text(self, file_url: str) -> str:
        downloaded = self.client.files.download(file_url)
        filename = downloaded.name
        content = downloaded.content
        file_extension = Path(filename).suffix.lower()
        return self.__extract_text(content, file_extension, filename)

    def __extract_text(self, file_content: bytes, file_extension: str, filename: str) -> str:
        """Extract text content based on file type."""
        try:
            if file_extension == '.txt':
                return file_content.decode('utf-8', errors='ignore')
            elif file_extension == '.pdf':
                pdf_bytes = io.BytesIO(file_content)
                with pdfplumber.open(pdf_bytes) as pdf:
                    pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
                return '\n'.join(pages)
            elif file_extension == '.csv':
                decoded_text_content = file_content.decode('utf-8', errors='ignore')
                csv_buffer = io.StringIO(decoded_text_content)
                df = pd.read_csv(csv_buffer)
                return df.to_markdown(index=False)
            elif file_extension in ['.html', '.htm']:
                decoded_html_content = file_content.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(decoded_html_content, features='html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator='\n', strip=True)
            else:
                return file_content.decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error extracting text from file: {e}")
            return ""
