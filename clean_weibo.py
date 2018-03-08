import pandas as pd
import re
import jieba.posseg as pseg
import nltk
import jieba
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
from PIL import Image, ImageDraw, ImageFont


import matplotlib.pyplot as plt


with open('./comments_data/comments2017.txt', 'r', encoding='gbk') as f:
    a = f.read()

comments_list = re.split(r'\d{4}-\d{1,2}-\d{1,2} \d{2}:\d{2}.+?\n', a)[:-1]
time_list = re.findall(r'\d{4}-\d{1,2}-\d{1,2} \d{2}:\d{2}.+?\n', a)
time_list = [re.findall(r'\d{4}-\d{1,2}-\d{1,2}', time)[0] for time in time_list]
comment_2017 = pd.DataFrame([time_list, comments_list]).T
comment_2017.columns = ['date', 'comment']

# with open('./comments_data/comments2018.txt', 'r', encoding='gbk') as f:
#     a = f.read()
# comments_list = re.split(r'\d{1,2}月\d{1,2}日\s*\d{2}:\d{2}.+?\n', a)[:-1]
# time_list = re.findall(r'\d{1,2}月\d{1,2}日\s*\d{2}:\d{2}.+?\n', a)
# time_list = [re.findall(r'\d{1,2}月\d{1,2}日', time)[0] for time in time_list]
# comment_2017 = pd.DataFrame([time, comments_list]).T
# comment_2017.columns = ['date', 'comment']


def screen_credit_card_comment():
    """
    :return:  招商银行信用卡的评论
    """
    credit_time, credit_comment = [], []
    for date, comment in zip(time_list, comments_list):
        if re.search(r'信用卡', comment):
            credit_comment.append(comment)
            credit_time.append(date)

    comments = pd.DataFrame([credit_time, credit_comment]).T
    comments.columns = ['date', 'comment']
    comments.to_excel('2017信用卡weibo.xlsx')


def words_bag(data):
    """
    :return: 高频词统计
    """
    stopwords = [line.rstrip() for line in open('./中文停用词库.txt', 'r', encoding='utf-8')]
    words_list = []
    comments_list = list(data['comment'])
    for comment in comments_list:
        def proc_comment():
            filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
            chinese_only = filter_pattern.sub('', comment)

            # 2. 结巴分词+词性标注
            # words_lst = pseg.cut(chinese_only)
            words_lst = jieba.cut(chinese_only)

            # 3. 去除停用词
            meaninful_words = []
            # for word, flag in words_lst:
            for word in words_lst:
                # if (word not in stopwords) and (flag == 'v'):
                # 也可根据词性去除非动词等
                if word not in stopwords:
                    meaninful_words.append(word)

            return ' '.join(meaninful_words)

        words_list.append(proc_comment())
    return words_list

words_list = words_bag(pd.read_excel('2017信用卡weibo.xlsx'))          # comments是处理后的
words_list = ' '.join(words_list)
fdisk = nltk.FreqDist(words_list.split(' '))
common_words_freqs = fdisk.most_common(30)
print('出现最多的{}个词是：'.format(30))


def word_cloud():
    """
    :return: 词云图
    """
    my_stopwords = ['展开','全文','微博','c2017','浏览器','新浪','网页','链接','iPhone','网页链接2017','weibo','来自',
                    '展开全文c','网页链接','发表了博文','生女儿的父母更幸福','一项研究显示','生儿子的马上攒钱买房','生儿子的父母',
                    '网友','观念已落伍',"重男轻女","在儿子17到30岁期间","幸福感明显低于生女儿的父母",
                    "高房价和男性过剩增加娶妻成本降低了父母的幸福感",'生女儿的计划买新车',"就是千古骂名"]

    backgroud_Image = plt.imread('back.jpg')
    wc = WordCloud(background_color='white',  # 设置背景颜色
                   mask=backgroud_Image,  # 设置遮罩图片,控制的是词云的形状，比如说松鼠形状的词云，云朵形状的等等，图清晰点比较好
                   max_words=1000,  # 设置最大现实的字数
                   stopwords=STOPWORDS.union(set(my_stopwords)),  # 设置停用词
                   font_path='MSYH.TTF',                          # 设置字体格式，如不设置显示不了中文
                   max_font_size=60,  # 设置字体最大值，会自动根据图片大写调整，不同图的60看起来不一样
                   min_font_size=2,    # 设的比较大的话，小的就不显示了
                   # random_state = 30,  # 设置有多少种随机生成状态，即有多少种布局方案,横的竖的分布
                   # scale=20    # 越大计算越慢，图的大小，不如让底图大点清晰点来得快
                   )
    # credit_card = pd.read_excel('2017信用卡weibo.xlsx')
    credit_card = pd.read_excel('comments.xlsx')
    text = ','.join(map(str, credit_card['comment']))
    wc.generate(text)
    image_colors = ImageColorGenerator(backgroud_Image)
    wc.recolor(color_func=image_colors)
    plt.imshow(wc)
    plt.axis('off')
    plt.savefig('wordcloud.png')
    # wc.to_file("wordcloud.png")
    plt.show()
word_cloud()


def screen_comment():
    """
    :return:  去除股票，板块。。。等跟客户体验没什么关系的微博
    """
    credit_time, credit_comment = [], []
    for date, comment in zip(time_list, comments_list):
        if not (re.search(r'官方客户端|发布了头条文章|发表了一篇转载博文|招聘|实习岗位', comment)
                or re.search(r'招商银行信用卡申请.+网页链接', comment)
                or re.search(r'看盘|大盘|财经要闻|盘面|股|财经视频|板块|sh600036|600036|茅台|上证|拉升|仓|投资策略', comment)) \
                and len(comment) > 5:
            credit_comment.append(comment)
            credit_time.append(date)

    comments = pd.DataFrame([credit_time, credit_comment]).T
    comments.columns = ['date', 'comment']
    comments.to_excel('2017weibo去除杂项.xlsx')
screen_comment()



