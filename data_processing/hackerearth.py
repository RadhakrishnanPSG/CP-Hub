from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from stopwords import remove_stopwords


def getHackerEarth():
    browser = webdriver.Chrome()
    documents = []
    links = []

    for i in range(0,82):
        
        browser.get("https://www.hackerearth.com/practice/problems/?limit=20&offset="+str(i))
        time.sleep(2)
        for j in range(1,21):
            try:
                elem2 = browser.find_element(By.XPATH, "/html/body/div/div/div[2]/div/div/div[4]/div[1]/div/div[1]/div/div["+ str(j) +"]/div[2]")
                link = browser.find_element(By.XPATH, "/html/body/div/div/div[2]/div/div/div[4]/div[1]/div/div[1]/div/div["+ str(j) +"]/div[2]/a")
            except:
                break

            documents.append(elem2.text.split() + ["hackerearth"])
            links.append([link.get_attribute('href')])

    browser.close()

    documents = remove_stopwords(documents)
    return documents, links



