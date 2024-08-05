from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import os
import time

target_crypts = ["bitcoin"]

start_date = "20130428"
end_date = "20180428"

current_dir = os.path.dirname(os.path.abspath(__file__))

option = Options()
option.add_experimental_option("prefs", {
    "download.default_directory": current_dir
})


def url(crypt, start_date, end_date):
    return f"https://coinmarketcap.com/currencies/{crypt}/historical-data/?start={start_date}&end={end_date}"

if __name__ == '__main__':
    driver = webdriver.Chrome(options=option)
    
    driver.get(url(target_crypts[0], start_date, end_date))
    time.sleep(1)

    download_btn = driver.find_element(By.XPATH, "//*[@id='__next']/div[2]/div/div[2]/div/div/div/div[2]/div/div[1]/div/button[2]/div[1]/div")
    download_btn.click()
    time.sleep(1)
    
    driver.quit()