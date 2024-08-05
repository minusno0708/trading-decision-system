from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import os
import time

target_crypts = ["bitcoin", "ethereum", "tether", "bnb", "solana", "usd-coin", "xrp", "toncoin", "dogecoin", "cardano"]

start_date = "20130428"
end_date = "20180428"

current_dir = os.path.dirname(os.path.abspath(__file__))
download_dir = "tmp"

option = Options()
option.add_experimental_option("prefs", {
    "download.default_directory": current_dir + "/" + download_dir,
})


def url(crypt, start_date, end_date):
    return f"https://coinmarketcap.com/currencies/{crypt}/historical-data/?start={start_date}&end={end_date}"

def make_dl_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def fetch_crypts():
    driver = webdriver.Chrome(options=option)
    
    for crypt in target_crypts:
        fetch_each_cript(driver, crypt)
    
    driver.quit()

def fetch_each_cript(driver, crypt):
    try:
        driver.get(url(crypt, start_date, end_date))
        time.sleep(1)

        download_btn = driver.find_element(By.XPATH, "//*[@id='__next']/div[2]/div/div[2]/div/div/div/div[2]/div/div[1]/div/button[2]/div[1]/div")
        download_btn.click()
        time.sleep(1)
    except Exception as e:
        print(f"{crypt}のデータ取得に失敗しました。")

if __name__ == '__main__':
    make_dl_dir(download_dir)

    fetch_crypts()

    