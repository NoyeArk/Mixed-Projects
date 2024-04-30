# coding=utf-8
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

print(-1)
urls = ('http://gkcx.eol.cn/soudaxue/queryProvince.html?page={}'.format(i) for i in range(1, 3))
print(0)
s = Service(executable_path=r'E:\programming\python\Python\PyCharm Community Edition 2022.2.2\edgedriver_win64\msedgedriver.exe')
driver = webdriver.Edge(service=s)
print(1)
driver.maximize_window()
print(2)
for url in urls:
    print('***************************')
    print ("正在访问{}".format(url))
    driver.get(url)
    time.sleep(3)
    data = driver.page_source
    print(data)
    soup = BeautifulSoup(data, 'lxml')
    grades = soup.find_all('tr')
    for grade in grades:
        if '<td>' in str(grade):
            print(grade.get_text())