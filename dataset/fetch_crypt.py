from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import os
import time
from datetime import datetime, timedelta

import glob

target_cryptos = ["bitcoin", "ethereum", "tether", "bnb", "solana", "usd-coin", "xrp", "toncoin", "dogecoin", "cardano"]

start_date = "20130528"
end_date = "20180528"

current_dir = os.path.dirname(os.path.abspath(__file__))
download_tmp_dir = "tmp"

option = Options()
#option.add_argument("--headless")
option.add_experimental_option("prefs", {
    "download.default_directory": current_dir + "/" + download_tmp_dir,
})

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

def url(crypto) -> str:
    return f"https://coinmarketcap.com/currencies/{crypto}/historical-data/?"

def make_dl_dir(dir) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)

def fetch_cryptos() -> None:
    driver = webdriver.Chrome(options=option)
    
    for crypto in target_cryptos:
        fetch_each_crypto(driver, crypto)
    
    driver.quit()

def fetch_each_crypto(driver, crypto) -> None:
    try:
        driver.get(url(crypto))
        time.sleep(3)

        # 日付けを選択
        choose_date(driver, start_date, end_date)

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

def concat_crypto_data() -> None:
    crypto_data = {}

    file_list = glob.glob(download_tmp_dir + "/*.csv")
    crypto_list = []
    for file in file_list:
        crypto_name = file.split("/")[1].split("_")[0]

        crypto_data[crypto_name] = read_crypto_file(file)
        crypto_list.append(crypto_name)
    
    with open("crypto_data.csv", "w") as f:
        labels = ["date"] + crypto_list
        f.write(",".join(labels) + "\n")

        date_list = list(crypto_data[crypto_list[0]].keys())

        for date in date_list:
            row = [date.strftime("%Y-%m-%d")]
            for crypto in crypto_list:
                row.append(crypto_data[crypto][date])
            
            f.write(",".join(row) + "\n")

def read_crypto_file(file) -> dict:
    result = {}

    with open(file, "r") as f:
        lines = f.readlines()
        columns = lines[0].replace("\n", "").replace("\ufeff", "").split(";")

        for row in lines[1:]:
            row = row.split(";")

            for i in range(len(columns)):
                if columns[i] == "timeOpen":
                    date = row[i].split("T")[0][1:]
                    date = datetime.strptime(date, "%Y-%m-%d")
                elif columns[i] == "close":
                    price = row[i]
                    result[date] = price
                    del date

    return result

if __name__ == '__main__':
    make_dl_dir(download_tmp_dir)

    #fetch_cryptos()
    concat_crypto_data()
    