import pandas as pd
import jieba
import re
from gensim import corpora, models
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

"""
    针对信用卡的微博，分正负面，看看词云还有LDA主题分析
"""


def screen_credit_card_comment(datafile):
    """
    :param datafile: weibo评论 dataframe
    :return: 招商银行信用卡的评论
    """
    comments_list = list(datafile['comment'])
    time_list = list(datafile['date'])
    credit_time, credit_comment = [], []
    for date, comment in zip(time_list, comments_list):
        if re.search(r'信用卡', comment) and not re.search(r'招商银行信用卡官方客户端|招商银行信用卡申请.+网页链接|【.*】|发表了博文', comment):
            credit_comment.append(comment)
            credit_time.append(date)

    comments = pd.DataFrame([credit_time, credit_comment]).T
    comments.columns = ['date', 'comment']
    comments = pd.merge(comments, datafile[['comment','category']], how='left')
    comments.to_excel('信用卡weibo.xlsx')
    return comments


def word_cloud(data, name):
    """
    :return: 词云图
    """
    my_stopwords = ['展开','全文','微博','c2017','浏览器','新浪','网页','链接','iPhone','网页链接2017','weibo','来自',
                    '展开全文c','网页链接','发表了博文','生女儿的父母更幸福','一项研究显示','生儿子的马上攒钱买房','生儿子的父母',
                    '网友','观念已落伍',"重男轻女","在儿子17到30岁期间","幸福感明显低于生女儿的父母",
                    "高房价和男性过剩增加娶妻成本降低了父母的幸福感",'生女儿的计划买新车',"就是千古骂名","招商银行",
                    "招商银行信用卡"]

    backgroud_Image = plt.imread('back.jpg')
    wc = WordCloud(background_color='white',  # 设置背景颜色
                   mask=backgroud_Image,  # 设置遮罩图片,控制的是词云的形状，比如说松鼠形状的词云，云朵形状的等等，图清晰点比较好
                   max_words=1500,  # 设置最大现实的字数
                   stopwords=STOPWORDS.union(set(my_stopwords)),  # 设置停用词
                   font_path='MSYH.TTF',                          # 设置字体格式，如不设置显示不了中文
                   max_font_size=40,  # 设置字体最大值，会自动根据图片大写调整，不同图的60看起来不一样
                   # min_font_size=2,    # 设的比较大的话，小的就不显示了
                   # random_state = 1800,  # 设置有多少种随机生成状态，即有多少种布局方案,横的竖的分布
                   # scale=20    # 越大计算越慢，图的大小，不如让底图大点清晰点来得快
                   )

    text = ','.join(map(str, data['comment']))
    wc.generate(text)
    image_colors = ImageColorGenerator(backgroud_Image)
    wc.recolor(color_func=image_colors)
    plt.imshow(wc)
    plt.axis('off')
    plt.savefig('wordcloud-'+name+'.png')
    # wc.to_file("wordcloud.png")
    plt.show()


class Topic:
    def __init__(self, data, stopwords, num_topics=3, num_words=5):
        self.data = data
        self.num_topics = num_topics
        self.stopwords = stopwords + ["招商银行", "信用卡", "说", "一次", "全文", "展开", "招行", "一个", "工商银行", "银行",
                      "建设银行", "招商", "建设", "之后", "链接", "网页"]
        self.data = self.sentence2word()
        self.dictionary = corpora.Dictionary(list(self.data["wordcut"]))  # 词典，key是词的id,value是词
        self.corpus = [self.dictionary.doc2bow(i) for i in self.data["wordcut"]]
        self.num_words = num_words

    def wordcut(self,line):
        """
        :param line: 一条微博
        :return: 去除非中文并分词
        """
        wordlist = [word for word in list(jieba.cut(re.compile('[^\u4E00-\u9FD5]+').sub('', line)))
                if word not in self.stopwords and len(word) > 1]
        return wordlist

    def sentence2word(self):
        self.data['wordcut'] = self.data["comment"].apply(self.wordcut)
        return self.data

    def tfidf_corpus(self):
        """
        :return: 每一个文档的feature是词语的tf-idf
        """
        corpus = models.TfidfModel(self.corpus)[self.corpus]  # [[（词id，词tf-idf),（词id，词tf-idf)...],[（词id，词tf-idf),...]]
        return corpus

    def lda(self):
        lda = models.LdaModel(self.corpus, num_topics=self.num_topics, id2word=self.dictionary)
        print("基于词频作feature的lda模型结果:\n{}".format(lda.print_topics(num_topics=3, num_words=self.num_words)))
        # topic_list = []
        # for i in range(self.num_topics):
        #     lda.print_topic(i)
        #     topic_list.append(lda.print_topic(i))

        # for i, doc in enumerate(lda[self.corpus]):
        #     if i < 5:
        #         print(doc)              # 可以看到每一个文档属于每个主题的概率

    def lda_tf_idf(self):
        """
        :return: tf_idf作为feature建lda模型
        """
        lda_tf_idf = models.LdaModel(self.tfidf_corpus(), num_topics=self.num_topics, id2word=self.dictionary)
        print("基于tf-idf作feature的lda模型结果:\n{}".format(lda_tf_idf.print_topics(num_topics=3, num_words=self.num_words)))
        doc_topic = [a for a in lda_tf_idf[self.tfidf_corpus()]]      # 每个文档的主题概率分布

    def lsi(self):
        lsi = models.lsimodel.LsiModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics)    # 初始化一个LSI转换
        doc_topic = lsi[self.corpus]
        print("基于词频作feature的lsi模型结果:\n{}".format(lsi.print_topics(num_topics=3, num_words=self.num_words)))

    def lsi_tf_idf(self):
        """
        :return:这个效果要好一点
        """
        lsi = models.lsimodel.LsiModel(corpus=self.tfidf_corpus(), id2word=self.dictionary, num_topics=self.num_topics)    # 初始化一个LSI转换
        doc_topic = lsi[self.tfidf_corpus()]                # 对其在向量空间进行转换
        print("基于tf-idf作feature的lsi模型结果:\n{}".format(lsi.print_topics(num_topics=3, num_words=self.num_words)))


if __name__ == '__main__':
    stopwords = [line.rstrip() for line in open('./中文停用词库.txt', 'r', encoding='utf-8')]
    comment = pd.read_excel("comments.xlsx")
    credit_data = screen_credit_card_comment(comment)  # 含信用卡关键字的非官方微博
    positive = credit_data[credit_data["category"].isin(["moderate positive", "very positive"])]
    negative = credit_data[credit_data["category"].isin(["moderate negative", "very negative"])]
    # word_cloud(positive, "positive")
    # word_cloud(negative, "negative")
    #
    # cluster_result = pd.read_excel("cluster_result.xlsx")
    # cluster_7 = cluster_result[cluster_result['cluster'] == 7]

    topic_analysis = Topic(negative, stopwords)
    topic_analysis.lda()
    topic_analysis.lda_tf_idf()
    topic_analysis.lsi()
    topic_analysis.lsi_tf_idf()

    topic_analysis = Topic(positive, stopwords)
    topic_analysis.lda()
    topic_analysis.lda_tf_idf()
    topic_analysis.lsi()
    topic_analysis.lsi_tf_idf()






