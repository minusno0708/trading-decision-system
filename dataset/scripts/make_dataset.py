from fetch_cryptos import fetch_cryptos
from concat_cryptos import concat_cryptos

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
download_tmp_dir = "tmp"

def make_dl_dir(dir) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    target_dir = parent_dir + "/" + download_tmp_dir

    make_dl_dir(target_dir)

    fetch_cryptos(target_dir)
    concat_cryptos(target_dir)