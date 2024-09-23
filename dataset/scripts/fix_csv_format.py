import sys
import csv

def read_csv(file):
    with open(file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            yield row

class WriteCSV:
    def __init__(self, file):
        self.file = file
        self.f = open(file, "w", encoding="utf-8")
        self.writer = csv.writer(self.f)
    def write(self, data):
        self.writer.writerow(data)
    def __del__(self):
        self.f.close()

if __name__ == "__main__":
    file_path = sys.argv[1]
    output_path = "../btc.csv"

    result = WriteCSV(output_path)

    for row in read_csv(file_path):
        result.write(row[0].replace('"', '').split(";"))