import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import pandas as pd

# 학회 페이지에서 카테고리 URL 추출
def get_category_urls(conference_url):
    response = requests.get(conference_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    a_tags = soup.select('#events-listing dt a')
    return [a.get('href') for a in a_tags]

# 각 카테고리 페이지에서 논문 상세 페이지 URL 추출
def get_paper_urls(category_url):
    response = requests.get(category_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    a_tags = soup.select('#series-home > table > tbody > tr > td > p > a')
    return [a.get('href') for a in a_tags]

# 논문 상세 페이지에서 정보 추출 (PDF 링크 포함)
def get_paper_info(paper_url):
    response = requests.get(paper_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.select_one('#title > h1 > a')
    title = title.text.strip() if title else None

    authors = [a.text.strip() for a in soup.select('#authors > p')]
    authors_str = ', '.join(authors)

    paper_number = soup.select_one('#paper_number p')
    paper_number = paper_number.text.strip() if paper_number else None

    abstract = soup.select_one('#abstract > p')
    abstract = abstract.text.strip() if abstract else None

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

# PDF 다운로드 함수
def download_pdf_from_url(pdf_url, filename):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ 다운로드 완료: {filename}")
    else:
        print(f"❌ 다운로드 실패 ({filename}) - 상태코드: {response.status_code}")

# 전체 실행
if __name__ == "__main__":
    base_url = 'https://aisel.aisnet.org'
    conference_url = urljoin(base_url, 'icis2021/')
    save_folder = 'icis2021_papers'
    os.makedirs(save_folder, exist_ok=True)

    # 모든 논문 메타데이터 저장용
    all_papers = []

    # 카테고리 순회
    category_urls = get_category_urls(conference_url)
    for cat_idx, category_url in enumerate(category_urls):
        print(f"\n📁 카테고리 {cat_idx + 1}/{len(category_urls)}: {category_url}")
        paper_urls = get_paper_urls(category_url)

        for paper_idx, paper_url in enumerate(paper_urls):
            full_paper_url = urljoin(base_url, paper_url)
            paper_info = get_paper_info(full_paper_url)

            filename = f"{cat_idx + 1:02}_{paper_idx + 1:03}.pdf"
            filepath = os.path.join(save_folder, filename)

            if paper_info['pdf_url']:
                download_pdf_from_url(paper_info['pdf_url'], filepath)
                paper_info['pdf_filename'] = filename
            else:
                paper_info['pdf_filename'] = None
                print(f"❌ PDF 링크 없음: {full_paper_url}")

            all_papers.append(paper_info)

    # DataFrame 생성 및 저장
    df = pd.DataFrame(all_papers)
    df.to_csv("icis2021_metadata.csv", index=False)
    print("✅ 메타데이터 저장 완료: icis2021_metadata.csv")