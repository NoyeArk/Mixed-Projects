import jieba
import wordcloud
import imageio
# https://www.cnblogs.com/denny402/p/5122594.html

bg=imageio.v2.imread('D:\\workSpace\\市场调研\\bg.png')
with open('D:/workSpace/市场调研/111.txt',encoding='utf-8') as f:
    t=f.read()

ls = jieba.lcut(t)
txt = " ".join(ls)

w=wordcloud.WordCloud(width=1000,height=700,#词云比例
font_path="msyh.ttc",colormap='prism',#字体&颜色
background_color='white',mask=bg)#背景色&模板
w.generate(txt)
w.to_file(r'D:/workSpace/市场调研/pic.png')