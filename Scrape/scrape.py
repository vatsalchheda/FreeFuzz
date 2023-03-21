from selenium import webdriver
from selenium.webdriver.common.by import By
import time
 
# WebDriver Chrome
driver = webdriver.Chrome()
 
# Target URL

driver.get('https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html')
# To load entire webpage
time.sleep(5)
elements = driver.find_elements(By.CLASS_NAME, "ant-menu-submenu-inline")
# print(elements)
list_apis = [] 
f = open("list_apis.txt", "a")
ex = open("errors.txt", "a") 
for el in elements:
    try:
        id = "//ul[@id='$1"+ el.text +"$Menu']/li"
        am = el.text + "."
        el.click()
        # f.write(el.text)
        subapis = [am + a.text for a in driver.find_elements(By.XPATH, id)]
        print(subapis)
        for m in subapis:
            f.write(m)
            f.write("\n")
        time.sleep(1)
        el.click()
        time.sleep(1)
    except Exception as e:
        print(f"An error occurred: with {e}")
        # ex.write(f"An error occurred: with {el.text}\n")
    # print(subapis)
    # subapis = [el.text + a for a in subapis]
    # print("*********" + el.text + "*******")
    # print(subapis)
# print(list_apis)
# Closing the driver
driver.close()
# f.close()
# ex.close()