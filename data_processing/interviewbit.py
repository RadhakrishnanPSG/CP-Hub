from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from stopwords import remove_stopwords


def getInterviewBit():

    browser = webdriver.Chrome()
    browser.get("https://www.interviewbit.com/coding-interview-questions/#")
    time.sleep(20)

    no_of_pagedowns = 50
    elem = browser.find_element_by_tag_name("body")
    elem2 = browser.find_element(By.XPATH, "/html/body/div[1]/div/div/div[3]/div/div/h1")
    elem2.click()


    while no_of_pagedowns:
        elem.send_keys(Keys.PAGE_DOWN)
        elem.send_keys(Keys.PAGE_DOWN)
        elem.send_keys(Keys.PAGE_DOWN)
        elem.send_keys(Keys.PAGE_DOWN)
        elem.send_keys(Keys.PAGE_DOWN)
        
        time.sleep(2)
        no_of_pagedowns-=1

    code = []
    i = 644
    while True:
        try:
            temp = browser.find_element(By.XPATH, "/html/body/div[1]/div/div/div[3]/div/div/div[4]/div/div/div/div[2]/div["+str(i)+"]")
            i-=1
            if i==0:
                break
            code.append(temp.text.split("\n"))
        except Exception as e:
            print(e)
            break            

        
    documents = []
    links = []

    for i in code:
        documents.append(i[0].split() + i[1].split() + ["interviewbit"])
        
        i[0] = i[0].replace("'"," ")
        i[0] = i[0].replace("."," ")
        i[0] = i[0].replace("-"," ")
        i[0] = i[0].lower()
        
        links.append(["https://www.interviewbit.com/problems/"+ "-".join(i[0].split(" ")) +"/"])

    
    documents = remove_stopwords(documents)
    return documents, links

