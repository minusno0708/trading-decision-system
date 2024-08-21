from datetime import datetime, timedelta

import glob

def concat_cryptos(download_dir) -> None:
    crypto_data = {}

    file_list = glob.glob(download_dir + "/*.csv")
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