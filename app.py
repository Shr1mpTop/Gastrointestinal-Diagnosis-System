from flask import Flask, render_template, redirect, url_for, request
import os
import socket
import time
from news_crawler import get_gastro_news

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/news')
def news():
    news_list = get_gastro_news()
    return render_template('news.html', news_list=news_list)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose')
def diagnose():
    # 检查端口是否可用
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        in_use = s.connect_ex(('localhost', 8501)) == 0
        
    if not in_use:
        os.system('streamlit run diagnose.py --server.port 8501 &')
        time.sleep(3)  # 等待服务启动
        
    return redirect('http://localhost:8501')

if __name__ == '__main__':
    # 生产环境建议使用 debug=False
    print("服务已启动，请访问 http://0.0.0.0:5000/news")
    print("本地访问地址：http://localhost:5000/news")
    print("远程访问地址：http://[服务器IP地址]:5000/news")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
