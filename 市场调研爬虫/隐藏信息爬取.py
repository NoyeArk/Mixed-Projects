# coding=utf-8
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

def get_tracks(distance, rate=0.6, t=0.2, v=0):
    """
    将distance分割成小段的距离
    :param distance: 总距离
    :param rate: 加速减速的临界比例
    :param a1: 加速度
    :param a2: 减速度
    :param t: 单位时间
    :param t: 初始速度
    :return: 小段的距离集合
    """
    tracks = []
    # 加速减速的临界值
    mid = rate * distance
    # 当前位移
    s = 0
    # 循环
    while s < distance:
        # 初始速度
        v0 = v
        if s < mid:
            a = 20
        else:
            a = -3
        # 计算当前t时间段走的距离
        s0 = v0 * t + 0.5 * a * t * t
        # 计算当前速度
        v = v0 + a * t
        # 四舍五入距离，因为像素没有小数
        tracks.append(round(s0))
        # 计算当前距离
        s += s0

    return tracks

def slide(driver):
    """滑动验证码"""
    # 切换iframe
    driver.switch_to.frame(1)
    # 找到滑块
    block = driver.find_element_by_xpath('//*[@id="tcaptcha_drag_button"]')
    # 找到刷新
    reload = driver.find_element_by_xpath('//*[@id="reload"]')
    while True:
        # 摁下滑块
        ActionChains(driver).click_and_hold(block).perform()
        # 移动
        ActionChains(driver).move_by_offset(180, 0).perform()
        # 获取位移
        tracks = get_tracks(30)
        # 循环
        for track in tracks:
            # 移动
            ActionChains(driver).move_by_offset(track, 0).perform()
        # 释放
        ActionChains(driver).release().perform()
        # 停一下
        time.sleep(2)
        # 判断
        if driver.title == "登录豆瓣":
            print("失败...再来一次...")
            # 单击刷新按钮刷新
            reload.click()
            # 停一下
            time.sleep(2)
        else:
            break

def move_slice1(driver, distance=280):
    "1、直接根据距离移动"
    print('move_slice1')
    elem = driver.find_element(by='class name', value='nc_iconfont btn_slide')
    print('ERROR!!!')
    print(elem)
    ActionChains(driver).click_and_hold(elem).perform()
    ActionChains(driver).move_by_offset(xoffset=distance,yoffset=0).perform()
    ActionChains(driver).release(elem).perform()

'''
<span id="nc_1_n1z" class="nc_iconfont btn_slide" style="left: 0px;"></span>
'''

if __name__ == "__main__":

    url = 'https://rate.taobao.com/user-rate-UMCx0vGILvm8S.htm?spm=a1z10.1-b-s.d4918101.1.5678f135FZeQ'
    print(0)
    s = Service(executable_path=r'E:\programming\python\Python\PyCharm Community Edition 2022.2.2\edgedriver_win64\msedgedriver.exe')
    driver = webdriver.Edge(service=s)
    print(1)
    driver.maximize_window()
    print(2)

    print('***************************')
    print ("正在访问{}".format(url))
    driver.get(url)
    # while(1):
    #     pass
    time.sleep(300)
    move_slice1(driver)
    data = driver.page_source
    print(data)
    # soup = BeautifulSoup(data, 'lxml')
    # grades = soup.find_all('tr')
    # for grade in grades:
    #     if '<td>' in str(grade):
    #         print(grade.get_text())