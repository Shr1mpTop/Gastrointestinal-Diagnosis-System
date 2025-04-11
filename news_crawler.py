import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
from flask import jsonify

class GastroNewsCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
        }
        self.sources = [
            {
                'name': '中华消化网',
                'url': 'http://www.chinadigestive.com/news/',
                'pattern': {
                    'container': 'div.news-list ul li',
                    'title': 'a',
                    'link': 'a@href',
                    'date': 'span.date'
                }
            },
            {
                'name': '医学界消化频道',
                'url': 'https://www.yxj.org.cn/digestive/news/',
                'pattern': {
                    'container': 'div.article-list div.item',
                    'title': 'h3 a',
                    'link': 'h3 a@href',
                    'date': 'div.meta span.time'
                }
            }
        ]

    def crawl_news(self):
        all_news = []
        for source in self.sources:
            try:
                response = requests.get(source['url'], headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for item in soup.select(source['pattern']['container']):
                    title_elem = item.select_one(source['pattern']['title'])
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    link = title_elem['href'] if 'href' in title_elem.attrs else ''
                    date_elem = item.select_one(source['pattern']['date'])
                    date = self.parse_date(date_elem.text.strip()) if date_elem else '未知时间'
                    
                    if not link.startswith('http'):
                        link = requests.compat.urljoin(source['url'], link)
                    
                    all_news.append({
                        'title': title,
                        'link': link,
                        'source': source['name'],
                        'date': date
                    })
                    
            except Exception as e:
                print(f"Error crawling {source['name']}: {str(e)}")
                continue
                
        # 按日期排序
        return sorted(all_news, key=lambda x: x['date'], reverse=True)[:15]

    def parse_date(self, date_str):
        try:
            # 处理中文日期格式
            date_str = re.sub(r'年|月', '-', date_str).replace('日', '')
            return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        except:
            return '未知时间'

def get_gastro_news():
    crawler = GastroNewsCrawler()
    return crawler.crawl_news()
