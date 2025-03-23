import requests
import fitz
import os

def download_pdf_from_url(url, filename="test.pdf"):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    else:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def get_pdf_text_from_url(url):
    temp_pdf = download_pdf_from_url(url)
    text = extract_text_from_pdf(temp_pdf)
    os.remove(temp_pdf)  # Clean up temporary file
    return text

# # ✅ 사용 예시
# pdf_url = "https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1292&context=icis2021"  # 여기에 실제 PDF URL 입력
# text = get_pdf_text_from_url(pdf_url)
# print(text[:500])  # 앞부분만 출력