import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# 연도 설정
# ==========================
year = 2025
start_page = 71      # ✅ 시작 페이지 지정
end_page = 80        # ✅ 종료 페이지 지정 (원하는 만큼 수정)
# ==========================

# 연도별 scope_id 매핑
scope_dict = {
    2021: '8db05028-a838-4e0f-911b-4ea544253c64',
    2022: '32d42543-f8b4-45fb-8c50-11a42cb8fe9a',
    2023: 'f36bf371-28c7-4427-bff7-718d2c995872',
    2024: '9d4848f6-1815-43f1-bcb5-be17e430d153',
    2025: 'c2af410a-9d5f-4f99-ad4e-626911b4e900'
}

scope_id = scope_dict.get(year)
if not scope_id:
    raise ValueError(f"❌ 연도 {year}에 대한 scope_id가 설정되어 있지 않습니다.")

# 기본 설정
BASE_URL = 'https://scholarspace.manoa.hawaii.edu'
BROWSE_BASE = f'{BASE_URL}/browse/dateissued?scope={scope_id}&bbm.page='
API_BASE = f'{BASE_URL}/server/api/core'
SAVE_DIR = f'HICS{year}_papers'
CSV_FILE = f'HICS{year}_metadata.csv'

def extract_paper_links_from_page(page_num: int) -> list:
    response = requests.get(f"{BROWSE_BASE}{page_num}")
    soup = BeautifulSoup(response.text, 'html.parser')
    a_tags = soup.select('a[href^="/items/"]')
    return [BASE_URL + a['href'] for a in a_tags if a.get('href')]

def extract_paper_metadata(url: str, save_dir: str = SAVE_DIR) -> pd.DataFrame:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.select_one('#main-content h1 span')
    title = title.text.strip() if title else None

    date = soup.select_one('ds-item-page-date-field span')
    date = date.text.strip() if date else None

    authors = soup.select('ds-metadata-representation-list ds-metadata-field-wrapper div div a')
    author_list = [a.text.strip() for a in authors if a.text.strip()] if authors else []

    abstract = soup.select_one('ds-item-page-abstract-field span')
    abstract = abstract.text.strip() if abstract else None

    keywords = soup.select_one('ds-generic-item-page-field:nth-child(4) span')
    keywords = keywords.text.strip().replace("\n", ", ") if keywords else None

    pdf_tag = soup.select_one('ds-file-download-link > a')
    pdf_url = pdf_tag['href'] if pdf_tag else None
    full_pdf_url = API_BASE + pdf_url if pdf_url and pdf_url.startswith("/") else pdf_url
    content_url = full_pdf_url.replace("/download", "/content") if full_pdf_url else None

    if content_url:
        try:
            pdf_response = requests.get(content_url)
            pdf_response.raise_for_status()
            os.makedirs(save_dir, exist_ok=True)
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", title or "unknown_title")
            filename = f"{safe_title}.pdf"
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(pdf_response.content)
        except Exception as e:
            print(f"[!] PDF 다운로드 실패: {e}")
    else:
        print("[!] PDF 링크를 찾을 수 없습니다.")

    return pd.DataFrame({
        'Title': [title],
        'Date': [date],
        'Authors': [', '.join(author_list)],
        'Abstract': [abstract],
        'Keywords': [keywords],
        'URL': [url],
        'PDF_URL': [full_pdf_url],
        'CONTENT_URL': [content_url]
    })

def main():
    print(f"🔍 HICSS {year} 논문 크롤링: 페이지 {start_page} ~ {end_page}")

    # 기존 파일이 있다면 데이터 누적
    if os.path.exists(CSV_FILE):
        result_df = pd.read_csv(CSV_FILE)
    else:
        result_df = pd.DataFrame()

    for page_num in tqdm(range(start_page, end_page + 1), desc="📄 페이지 진행"):
        paper_links = extract_paper_links_from_page(page_num)

        if not paper_links:
            print(f"[!] 페이지 {page_num}에 논문 링크가 없습니다.")
            continue

        page_metadata = []
        for link in tqdm(paper_links, desc=f"📥 Page {page_num} 논문 다운로드", leave=False):
            df = extract_paper_metadata(link)
            if df is not None and not df.empty:
                page_metadata.append(df)
            else:
                print(f"[!] 링크 {link}에서 메타데이터 추출 실패")

            time.sleep(1)  # 논문 간 대기

        if page_metadata:
            # 페이지별 데이터 누적 및 저장
            page_df = pd.concat(page_metadata, ignore_index=True)
            result_df = pd.concat([result_df, page_df], ignore_index=True)
            result_df.to_csv(CSV_FILE, index=False)
        else:
            print(f"[!] 페이지 {page_num}에서 메타데이터가 없습니다.")

        time.sleep(3)  # 페이지 간 대기

    print(f"\n✅ 지정된 페이지({start_page}-{end_page})의 논문 정보가 '{CSV_FILE}'에 누적 저장되었습니다.")

if __name__ == '__main__':
    main()


# 71~