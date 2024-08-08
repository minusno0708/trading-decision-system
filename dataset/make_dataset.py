from fetch_cryptos import fetch_cryptos
from concat_cryptos import concat_cryptos

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
download_tmp_dir = "tmp"

def make_dl_dir(dir) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    make_dl_dir(download_tmp_dir)

    fetch_cryptos(current_dir + "/" + download_tmp_dir)
    concat_cryptos(current_dir + "/" + download_tmp_dir)