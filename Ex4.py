from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from bs4 import BeautifulSoup

# Exercise 4
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://old.reddit.com/")
content = driver.page_source
soup = BeautifulSoup(content, features="html.parser")
tabs = soup.find_all("div", class_="linkflair")

for tab in tabs:
    subreddit = tab.select("a.subreddit")[0].get_text()
    upvotes = tab.select("div.score.unvoted")[0].get_text()
    post_title = tab.select("p.title a.title")[0].get_text()
    time = tab.select("time")[0]["title"]
    print(subreddit, time, upvotes, post_title)

driver.quit()
