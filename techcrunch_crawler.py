import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# 저장용 리스트
data = []

# 페이지 순회 (1~232)
for page in range(1, 2):
    url = f'https://techcrunch.com/category/artificial-intelligence/page/{page}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 기사 목록 선택
    articles = soup.select('.wp-block-post-template.is-layout-flow.wp-block-post-template-is-layout-flow > li')

    for article in articles:
        # 'AI' 태그가 있는 기사만
        label_link = article.select_one('div > div > div > div > a')
        if label_link and label_link.text.strip() == 'AI':
            # 기사 링크 추출
            title_link = article.select_one('div > div > div > h3 > a')
            if title_link and title_link.get('href'):
                article_url = title_link.get('href')
                response2 = requests.get(article_url)
                soup2 = BeautifulSoup(response2.text, 'html.parser')

                # 제목
                head = soup2.select_one('.article-hero__title.wp-block-post-title')
                title = head.get_text(strip=True) if head else None

                # 날짜
                date_tag = soup2.select_one('.wp-block-post-date > time')
                date = date_tag.get('datetime') if date_tag else None

                # 본문
                content_tags = soup2.select(
                    '.entry-content.wp-block-post-content.is-layout-constrained.wp-block-post-content-is-layout-constrained > p'
                )
                content = "\n".join(p.get_text(strip=True) for p in content_tags) if content_tags else None

                # 키워드
                keywords_tag = soup2.select_one('.wp-block-tc23-post-relevant-terms > div')
                keywords = keywords_tag.get_text(strip=True) if keywords_tag else None

                # 누적 저장
                data.append({
                    'title': title,
                    'date': date,
                    'content': content,
                    'keywords': keywords,
                    'url': article_url
                })

                # 예의상 딜레이
                time.sleep(0.5)

# 데이터프레임으로 변환
df = pd.DataFrame(data)
print(f"{len(df)}개의 기사를 수집했습니다.")

# 필요시 저장
# df.to_csv('techcrunch_ai_articles.csv', index=False)