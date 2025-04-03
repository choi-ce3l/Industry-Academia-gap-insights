import requests
import fitz  # PyMuPDF
import os
import pandas as pd
import re

def extract_keywords_section(text: str) -> str:
    pattern = r"keywords[s]?:[\s\S]*?(?=\n\s*(?:\d*\.*\s*)?Introduction)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        extracted = match.group(0)
        # "keywords:" 제거하고 줄바꿈/공백 정리
        cleaned = re.sub(r"(?i)keywords[s]?:", "", extracted)  # 대소문자 무시하고 키워드 제거
        return " ".join(cleaned.split())  # 줄바꿈 제거 + 공백 정리
    else:
        return ""

def get_keywords_from_url(url, max_pages=5):
    temp_pdf = download_pdf_from_url(url)
    df_pages = extract_text_per_page(temp_pdf, max_pages=max_pages)
    os.remove(temp_pdf)

    # 앞 5페이지 전체 텍스트
    full_text = "\n".join(df_pages['text'].tolist())
    return extract_keywords_section(full_text)

def download_pdf_from_url(url, filename="temp.pdf"):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    else:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")

def extract_text_per_page(pdf_path, max_pages=5):
    doc = fitz.open(pdf_path)
    data = []
    for page_num in range(min(max_pages, len(doc))):
        page = doc.load_page(page_num)
        text = page.get_text()
        data.append({"page_number": page_num + 1, "text": text})
    doc.close()
    return pd.DataFrame(data)

def get_pdf_text_dataframe_from_url(url, max_pages=5):
    temp_pdf = download_pdf_from_url(url)
    df = extract_text_per_page(temp_pdf, max_pages=max_pages)
    os.remove(temp_pdf)
    return df

# ✅ 사용 예시
pdf_url = "https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1292&context=icis2021"
df_text = get_pdf_text_dataframe_from_url(pdf_url, max_pages=5)

# 미리보기
print(df_text.head())

# 저장 (원하는 경우)
# df_text.to_csv("pdf_text_output.csv", index=False)
