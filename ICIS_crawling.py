import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import pandas as pd

class ICISPaperScraper:
    def __init__(self, year):
        self.year = str(year)
        self.base_url = 'https://aisel.aisnet.org'
        self.conference_url = urljoin(self.base_url, f'icis{self.year}/')
        self.save_folder = f'icis{self.year}_papers'
        self.metadata_file = f'icis{self.year}_metadata.csv'
        os.makedirs(self.save_folder, exist_ok=True)
        self.all_papers = []

    def get_category_urls(self):
        response = requests.get(self.conference_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        a_tags = soup.select('#events-listing dt a')
        return [urljoin(self.conference_url, a.get('href')) for a in a_tags]

    def get_paper_urls(self, category_url):
        response = requests.get(category_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        a_tags = soup.select('#series-home > table > tbody > tr > td > p > a')
        return [urljoin(category_url, a.get('href')) for a in a_tags]

    def get_paper_info(self, paper_url):
        response = requests.get(paper_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        title_tag = soup.select_one('#title > h1 > a')
        title = title_tag.text.strip() if title_tag else None

        authors = [a.text.strip() for a in soup.select('#authors > p')]
        authors_str = ', '.join(authors) if authors else None

        paper_number_tag = soup.select_one('#paper_number p')
        paper_number = paper_number_tag.text.strip() if paper_number_tag else None

        abstract_tag = soup.select_one('#abstract > p')
        abstract = abstract_tag.text.strip() if abstract_tag else None

        pdf_tag = soup.select_one('#pdf')
        pdf_url = urljoin(paper_url, pdf_tag.get('href')) if pdf_tag else None

        return {
            'title': title,
            'authors': authors_str,
            'paper_number': paper_number,
            'abstract': abstract,
            'pdf_url': pdf_url,
            'paper_url': paper_url
        }

    def download_pdf(self, pdf_url, filename):
        try:
            response = requests.get(pdf_url, stream=True)
            if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print(f"✅ 다운로드 완료: {filename}")
            else:
                print(f"❌ PDF 오류 또는 형식 이상: {filename} - 상태코드 {response.status_code}")
        except Exception as e:
            print(f"❌ 예외 발생 ({filename}): {e}")

    def run(self):
        category_urls = self.get_category_urls()
        print(f"🔗 총 {len(category_urls)}개 카테고리 발견")

        for cat_idx, category_url in enumerate(category_urls):
            print(f"\n📁 카테고리 {cat_idx + 1}/{len(category_urls)}: {category_url}")
            paper_urls = self.get_paper_urls(category_url)

            for paper_idx, paper_url in enumerate(paper_urls):
                print(f"   📄 논문 {paper_idx + 1}/{len(paper_urls)}: {paper_url}")
                paper_info = self.get_paper_info(paper_url)

                filename = f"{cat_idx + 1:02}_{paper_idx + 1:03}.pdf"
                filepath = os.path.join(self.save_folder, filename)

                if paper_info['pdf_url']:
                    self.download_pdf(paper_info['pdf_url'], filepath)
                    paper_info['pdf_filename'] = filename
                else:
                    paper_info['pdf_filename'] = None
                    print(f"❌ PDF 링크 없음: {paper_url}")

                self.all_papers.append(paper_info)

        df = pd.DataFrame(self.all_papers)
        df.to_csv(self.metadata_file, index=False)
        print(f"\n✅ 메타데이터 저장 완료: {self.metadata_file}")
        return df
'''    
# 사용 예시
scraper = ICISPaperScraper(year=2021)
df = scraper.run()
df.head()
'''