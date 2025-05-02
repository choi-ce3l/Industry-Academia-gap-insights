import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 1️⃣ 크롬 설정
options = Options()
# options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 10)

# 2️⃣ 시작 페이지 URL 목록 만들기
base_url = "https://misq.umn.edu/contents-{}-{}/"
start_urls = [base_url.format(vol, issue) for vol in range(45, 50) for issue in range(1, 5)]

# ✅ 볼륨-연도 매핑
volume_year_map = {
    45: "2021",
    46: "2022",
    47: "2023",
    48: "2024",
    49: "2025"
}

# 3️⃣ CSV 저장 준비
with open('misq.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['title', 'author', 'year', 'abstract', 'keywords', 'url'])

    # 4️⃣ 시작 페이지 순회
    for start_url in start_urls:
        print(f"📖 시작 페이지 접속: {start_url}")
        driver.get(start_url)

        # 현재 볼륨 넘버 추출
        try:
            vol_num = int(start_url.split('-')[1])
            year = volume_year_map.get(vol_num, "Unknown")
        except:
            year = "Unknown"

        # 상세 페이지 URL 수집
        article_links = driver.find_elements(By.CSS_SELECTOR, "a[href$='.html']")
        article_urls = list({a.get_attribute("href") for a in article_links})

        for url in article_urls:
            driver.get(url)
            try:
                title = wait.until(EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="maincontent"]/div/div/div/div/div/h1'))).text

                # abstract: div/p 또는 div
                try:
                    abstract = driver.find_element(By.XPATH, '//*[@id="maincontent"]/div/div/div/div/p').text
                except:
                    abstract = driver.find_element(By.XPATH, '//*[@id="maincontent"]/div[2]/div/div[3]/div[2]').text

                author = driver.find_element(By.XPATH, '//*[@id="maincontent"]/div[2]/div/div[4]/div[2]/div/table/tbody/tr[1]/td[2]').text

                try:
                    keywords = driver.find_element(By.XPATH, '//*[@id="maincontent"]/div[2]/div/div[4]/div[2]/div/table/tbody/tr[5]/td[2]').text
                except:
                    keywords = "None"

                writer.writerow([title, author, year, abstract, keywords, url])
                print(f"✅ 저장 완료: {title}")

            except Exception as e:
                print(f"❌ {url} 에서 오류 발생:", e)

# 5️⃣ 종료
driver.quit()
print("📄 모든 논문 크롤링 완료! 저장 파일: misq.csv")