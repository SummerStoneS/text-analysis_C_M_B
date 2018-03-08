from bosonnlp import BosonNLP
from bosonnlp.exceptions import HTTPError
from threading import Thread
from multiprocessing import Event
from queue import Queue
import re
from collections import defaultdict
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import datetime


class BosonThread(Thread):
    def __init__(self, token, qin: Queue, qout: Queue, event: Event):
        super(BosonThread, self).__init__()
        self.api = BosonNLP(token)
        self.qin = qin
        self.qout = qout
        self.stop_event = event

    def run(self):
        while not self.qin.empty():
            if self.stop_event.is_set():
                return
            time, comment = self.qin.get()
            try:
                sentiment = self.api.sentiment(comment, model='weibo')
                self.qout.put([time, comment, sentiment])
            except Exception as e:
                print(e)
                sleep(60)


def cal_sentiment(data):
    """
    :return: 用Boson提供的接口计算情感的分数，元组第一个表示positive的概率，第二个表示negative的概率
    """
    qin = Queue()
    qout = Queue()
    stop_event = Event()
    threads = [BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),

               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               BosonThread('', qin, qout, stop_event),
               ]

    sentiment_list = []
    for task in zip(list(data['date']), list(data['comment'])):
        qin.put(task)

    for thread in threads:
        thread.start()

    try:
        while not qin.empty():
            flag = False
            while not qout.empty():
                sentiment = qout.get()
                sentiment_list.append(sentiment)
                flag = True
            if flag:
                pd.DataFrame(sentiment_list).to_excel("sentiment_result.xlsx")
            sleep(10)
    except KeyboardInterrupt:           # 如果
        stop_event.set()
        for thread in threads:
            thread.join()

data = pd.read_excel('2017weibo_rest.xlsx')
cal_sentiment(data)


# 统计每天评论条数，正向数，负向数，中性数
# dict-key: 时间；dict-value:每条微博sentiment构成的list
comments = pd.read_excel('sentiment_result.xlsx')
sentiments_dict = defaultdict(list)
for time, comment in zip(comments['date'], comments['sentiment']):
    sentiments_dict[str(time)].append(comment)

summary = []
for key, value in sentiments_dict.items():
    time = key
    count = len(value)              # 这一天的评论数
    negative = 0
    positive = 0
    neutral = 0
    for sentiment in value:
        if eval(sentiment)[0][1] > 0.57:
            negative += 1
        elif eval(sentiment)[0][0] > 0.57:
            positive += 1
        else:
            neutral += 1
    summary.append({'date': time, 'count': count, 'negative': negative, 'positive': positive})

summary = pd.DataFrame(summary)
summary['date'] = pd.to_datetime(summary['date'])
summary = summary[summary['count'] > 11]
summary.sort_values(by='date', inplace=True)
summary['positive_percent'] = summary['positive']/summary['count']
summary['date'] = summary['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
summary.to_excel('summary_0307.xlsx')

# summary.index = summary['date']
# del summary['date']
# summary[['count','negative','positive']].plot()
# plt.show()
#
# summary[['positive_persent']].plot(kind='bar')
# plt.show()

#
# plt.plot(summary[['count','negative','positive']])
# plt.xlabel(list(summary['date']))
# plt.legend(['count','negative','positive'])
# plt.show()
#
# plt.plot(summary['positive_percent'])
# plt.xlabel(list(pd.to_datetime(summary['date'])))
# plt.title('positive percent of daily comments')
# plt.show()


def pie_chart():
    """
    :return: 半年里的weibo 情绪属于很消极，一般消极，一般积极，很积极，中性的分布
    """
    comments = pd.read_excel('sentiment_result.xlsx')
    def sentiment_category(x):
        if eval(x)[0][1] > 0.9:        # 非常消极情绪
            return 'very negative'
        elif eval(x)[0][1] > 0.57:       # normal negative
            return 'moderate negative'
        elif eval(x)[0][0] > 0.9:      # 非常积极情绪
            return 'very positive'
        elif eval(x)[0][0] > 0.57:
            return 'moderate positive'
        else:
            return 'neutral'

    comments['category'] = comments['sentiment'].apply(sentiment_category)
    pie_data = comments[['category','sentiment']].groupby('category').count()
    pie_data['percent'] = pie_data['sentiment']/len(comments)
    pie_data.to_excel('pie.xlsx')






"""
    查询使用额度
"""
import requests
HEADERS = {'X-Token': 'sTTOsPHu.24178.XXaXRsUkpp2V'}
RATE_LIMIT_URL = 'http://api.bosonnlp.com/application/rate_limit_status.json'
result = requests.get(RATE_LIMIT_URL, headers=HEADERS).json()



