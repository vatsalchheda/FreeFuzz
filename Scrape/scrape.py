import requests
from bs4 import BeautifulSoup

page = requests.get('https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html')

soup = BeautifulSoup(page.text, 'html.parser')

links = soup.findAll(class_='reference internal')

final = []
for link in links:
    elems = link['href'].split('/')
    if 'api' in elems:
        index = elems.index('api')
        final.append(".".join(elems[index+1:])[:-8])

f = open("paddle.txt", "a")
for l in final:
    f.write(l)
    f.write("\n")
f.close()