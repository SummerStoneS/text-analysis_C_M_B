import selenium.webdriver
from time import sleep
import re
import datetime


INDEX_URL = "https://weibo.com/login"       # 微博登陆
# 搜索完关键词的url，通过高级搜索可以看到下面这个带时间的url,根据终止时间共有50页，开始时间一般没什么用，可以设为终止时间前一个月
URL_PATTERN = "http://s.weibo.com/weibo/%25E6%258B%259B%25E5%2595%2586%25E9%2593%25B6%25E8%25A1%258C&typeall=1&suball=1&timescope=custom:{start_dt}:{end_dt}&page={page}"

class Weibo:
    def __init__(self, driver, username, password):
        self.driver = getattr(selenium.webdriver, driver)()         # selenium.webdriver.Chrome()
        self.username = username
        self.password = password

    def login(self):
        self.driver.get(INDEX_URL)
        self.driver.implicitly_wait(1)
        self.driver.maximize_window()
        user_input = self.driver.find_element_by_id("loginname")

        # logging.info user_input.get_attribute('action-data')
        user_input.click()
        user_input.clear()
        user_input.send_keys(self.username)

        passwd_input = self.driver.find_element_by_name("password")
        passwd_input.click()
        passwd_input.clear()
        # logging.info passwd_input
        passwd_input.send_keys(self.password)

        # submit_button = self.driver.find_element_by_xpath("//div[@id='pl_login_form']/div/div[3]/div[6]/a")
        # submit_button.click()
        # sleep(5)
        input("input verification code")        # 需要手动输入验证码

    def view_page(self, start_dt, end_dt, page):
        self.driver.get(URL_PATTERN.format(start_dt=start_dt, end_dt=end_dt, page=page))
        elements = self.driver.find_elements_by_xpath('//div[@id="pl_weibo_direct"]/div/div/div/div/div/div/dl/div/div[3]/div/p') # 每条微博内容的xpath
        for ele in elements:
            datee = ele.find_element_by_xpath('//div[@class="feed_from W_textb"]').text  # 时间不准确，实际时间在微博内容外面，这个只是每页第一条微博的时间
            yield ele.text,datee
        sleep(12)


if __name__ == "__main__":

    user_name = ''
    password = ''
    weibo = Weibo("Chrome", user_name, password)
    weibo.login()

    def get_50_pages(start,end):
        """
        :param start:
        :param end:
        :return: 给定起止时间，抓50页的微博内容
        """
        with open("comments5.txt", "a",encoding='utf-8') as f:
            for page in range(50):
                for comment,datee in weibo.view_page(start, end, page+1):
                    print(datee)
                    if re.search(re.compile("招商银行"), comment):      # 有的时候页面刷出来没找到招商银行，会出现别的不相关微博，就不能要
                        f.write(comment+datee + "\n")
        return datee

    # datee = get_50_pages('2017-10-28','2017-11-16')
    # match = re.match(r"\d月\d日", datee)
    # date_number = re.findall(r"\d", match.group())
    # date_list = ['{:0>2}'.format(number) for number in date_number]       # 2018年后的日期格式
    # recent_date = '-'.join(['2018']+date_list)
    date_start = '2017-06-15'
    date_end2 = '2017-07-15'
    # while int(date_number[0])>7:
    while datetime.datetime.strptime(date_end2, '%Y-%m-%d') >= datetime.datetime(2017, 1, 1):
        datee = get_50_pages(date_start, date_end2)
        recent_date = re.findall(re.compile(r'[0-9]+-[0-9]+-[0-9]+'), datee)[0]     #捕获最后一页第一条微博的时间，然后-1天，作为新的终止时间，再抓
        date_end = datetime.datetime.strptime(recent_date, '%Y-%m-%d')
        date_end2 = date_end-datetime.timedelta(1)
        date_start = date_end-datetime.timedelta(30)
        date_start = datetime.datetime.strftime(date_start,'%Y-%m-%d')
        date_end2 = datetime.datetime.strftime(date_end2, '%Y-%m-%d')




