from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import os
import time
from datetime import datetime, timedelta

import glob

target_cryptos = ["bitcoin", "ethereum", "tether", "bnb", "solana", "usd-coin", "xrp", "toncoin", "dogecoin", "cardano"]

start_date = "20130428"
end_date = "20180428"

current_dir = os.path.dirname(os.path.abspath(__file__))
download_tmp_dir = "tmp"

option = Options()
option.add_argument("--headless")
option.add_experimental_option("prefs", {
    "download.default_directory": current_dir + "/" + download_tmp_dir,
})


def url(crypto, start_date, end_date) -> str:
    return f"https://coinmarketcap.com/currencies/{crypto}/historical-data/?start={start_date}&end={end_date}"

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
        driver.get(url(crypto, start_date, end_date))
        time.sleep(1)

        download_btn = driver.find_element(By.XPATH, "//*[@id='__next']/div[2]/div/div[2]/div/div/div/div[2]/div/div[1]/div/button[2]/div[1]/div")
        download_btn.click()
        time.sleep(1)
        print(f"{crypto}のデータを取得しました。")
    except Exception as e:
        print(f"{crypto}のデータ取得に失敗しました。")

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

    fetch_cryptos()
    concat_crypto_data()
    