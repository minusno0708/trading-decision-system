from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import os
import time

import glob

target_cryptos = ["bitcoin", "ethereum", "tether", "bnb", "solana", "usd-coin", "xrp", "toncoin", "dogecoin", "cardano"]

start_date = "20130528"
end_date = "20180528"

month_dict = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec"
}

def fetch_cryptos(download_dir) -> None:
    option = Options()
    #option.add_argument("--headless")
    option.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
    })

    driver = webdriver.Chrome(options=option)
    
    for crypto in target_cryptos:
        fetch_each_crypto(driver, crypto)
    
    driver.quit()

def url(crypto) -> str:
    return f"https://coinmarketcap.com/currencies/{crypto}/historical-data/?"

def fetch_each_crypto(driver, crypto) -> None:
    try:
        driver.get(url(crypto))
        time.sleep(3)

        try:
            driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div/div[2]/div/div/div/div[2]/div/div[1]/h1').click()
            time.sleep(1)
            driver.find_element(By.XPATH, '//*[@id="onetrust-reject-all-handler"]').click()
            time.sleep(1)
        except:
            pass

        # 日付けを選択
        choose_date(driver, start_date, end_date)
    
        while True:
            try:
                driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div/div[2]/div/div/div/div[2]/div/p[1]/button').click()
                time.sleep(1)
            except:
                break

        download_btn = driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div/div[2]/div/div/div/div[2]/div/div[1]/div/button[2]/div[1]/div')
        download_btn.click()
        time.sleep(1)
        print(f"{crypto}のデータを取得しました。")
    except Exception as e:
        print(f"{crypto}のデータ取得に失敗しました。")

def choose_date(driver, start_date, end_date) -> None:
    start_date_list = [int(start_date[:4]), int(start_date[4:6]), int(start_date[6:])]
    end_date_list = [int(end_date[:4]), int(end_date[4:6]), int(end_date[6:])]

    date_btn = driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div/div[2]/div/div/div/div[2]/div/div[1]/div/button[1]/div[1]/div')
    date_btn.click()

    span_btn = driver.find_element(By.XPATH, '//*[@id="tippy-1"]/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[1]/div[1]/span[2]')
    span_btn.click()
    span_btn.click()
    time.sleep(1)

    year_area = driver.find_element(By.CLASS_NAME, 'yearpicker')
    month_area = driver.find_element(By.CLASS_NAME, 'monthpicker')
    day_area = driver.find_element(By.CLASS_NAME, 'react-datepicker__month')

    while (start_date_list[0] < int(span_btn.text.split("-")[0])):
        switch_btn = driver.find_element(By.XPATH, '//*[@id="tippy-1"]/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[1]/div[1]/span[1]')
        switch_btn.click()
        time.sleep(1)

    year_btn = year_area.find_element(By.XPATH, "//span[contains(text(), '" + str(start_date_list[0]) + "')]")
    year_btn.click()
    time.sleep(1)

    month_btn = month_area.find_elements(By.XPATH, "//span[contains(text(), '" + month_dict[start_date_list[1]] + "')]")
    month_btn[-1].click()
    time.sleep(1)

    day_btn = day_area.find_elements(By.XPATH, "//div[contains(text(), '" + str(start_date_list[2]) + "')]")
    if len(day_btn) == 1:
        day_btn[0].click()
    else:
        if start_date_list[2] < 15:
            day_btn[0].click()
        else:
            day_btn[1].click()
    time.sleep(1)

    span_btn.click()
    span_btn.click()
    time.sleep(1)

    while (end_date_list[0] > int(span_btn.text.split("-")[1])):
        switch_btn = driver.find_element(By.XPATH, '//*[@id="tippy-1"]/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[1]/div[1]/span[3]')
        switch_btn.click()
        time.sleep(1)

    year_btn = year_area.find_element(By.XPATH, "//span[contains(text(), '" + str(end_date_list[0]) + "')]")
    year_btn.click()
    time.sleep(1)
    month_btn = month_area.find_elements(By.XPATH, "//span[contains(text(), '" + month_dict[end_date_list[1]] + "')]")
    month_btn[-1].click()
    time.sleep(1)
    day_btn = day_area.find_elements(By.XPATH, "//div[contains(text(), '" + str(end_date_list[2]) + "')]")
    if len(day_btn) == 1:
        day_btn[0].click()
    else:
        if start_date_list[2] < 15:
            day_btn[0].click()
        else:
            day_btn[1].click()
    time.sleep(1)

    continue_btn = driver.find_element(By.XPATH, '//*[@id="tippy-1"]/div/div[1]/div/div/div[2]/span/button')
    continue_btn.click()
    time.sleep(1)
