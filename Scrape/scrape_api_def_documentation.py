import os
import shutil
import time
import multiprocessing as mp
import requests
import bs4
from selenium import webdriver
from selenium.webdriver.common.by import By

path = './api_documentation_code/'

class dummy:
    text = ""
    def __init__(self):
        text = ""

def process(url, driver):
    driver.get(url)
    time.sleep(3)
    try:
        api_def = driver.find_elements(By.XPATH, "/html/body/div[5]/div/div[7]/div/div[4]/div[4]/div/div/div/dl/dt")[0]
    except:
        api_def = driver.find_elements(By.XPATH, "/html/body/div[4]/div/div[7]/div/div[4]/div[4]/div/div/div/dl/dt")[0]
    try:    
        example = driver.find_elements(By.XPATH, "/html/body/div[5]/div/div[7]/div/div[4]/div[4]/div/div/div/dl/dd/div")[-1]
    except:
        try:
            example = driver.find_elements(By.XPATH, "/html/body/div[4]/div/div[7]/div/div[4]/div[4]/div/div/div/dl/dd/div")[-1]
        except:
            print(f"No example for {url[58:-8]}")
            example = dummy()
    finally:
        try:
            if type(example).__name__ != 'dummy':
                example_file = open(path + url[58:-8].replace('/','_') + ".py","w+")
                example_file.write(example.text)
                example_file.close()

            return api_def.text[:-9].replace(" ","")
        except:
            print(f"API Definition Issue for {url[58:-8]}")
  
def scrape_page(url):
    # Scrape the page and return the data
    driver = webdriver.Chrome("E:/UIUC/Spring 2023/CS 527/FreeFuzz/Scrape/chromedriver.exe")
    driver.get(url)
    api_def = process(url, driver)
    driver.close()
    return api_def
  
def scrape_pages_mp(urls):
    with mp.Pool(10) as p:
        results = p.map(scrape_page, urls)
    return results
if __name__ == '__main__':
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)
    page = requests.get('https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html')
    soup = bs4.BeautifulSoup(page.text, 'html.parser')
    base_url = 'https://www.paddlepaddle.org.cn'
    links = soup.findAll(class_='reference internal')
    api_documentation_links = []

    final = []
    for link in links:
        elems = link['href'].split('/')
        if 'api' in elems:
            index = elems.index('api')
            final.append(".".join(elems[index+1:])[:-8])
            api_documentation_links.append(base_url + link['href'])

    # Test the multiprocessed scraper
    start = time.time()
    data = scrape_pages_mp(api_documentation_links[0:10])
    end = time.time()
    f = open("api_def.txt", "w+")
    for l in data:
        f.write(l)
        f.write("\n")
    f.close()
    print(f"Time taken for multiprocessed scraper: {end - start} seconds")