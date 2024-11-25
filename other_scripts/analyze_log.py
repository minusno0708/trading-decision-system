import sys
import numpy as np

class LogAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.matrics_name = self.get_matric_name()
        self.matrics = {}

        print(self.matrics_name)

        for target_matric in self.matrics_name:
            self.matrics[target_matric] = self.read_matric(target_matric)

    def get_matric_name(self):
        matrics_list = []
        backtest_flag = False
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if ":" in line and backtest_flag:
                    target_matric = line.split(":")[0].strip()
                    matrics_list.append(target_matric)
                elif "backtest" in line:
                    backtest_flag = True
                elif "Evaluation Done" in line:
                    break
        return matrics_list

    def read_matric(self, target_matric):
        matric_list = np.array([])
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if target_matric == line.split(":")[0].strip():
                    matric = float(line.split(":")[1].strip())
                    matric_list = np.append(matric_list, matric)

        return matric_list

    def matric_len(self, target_matric):
        return len(self.matrics[target_matric])

    def matric_ave(self, target_matric):
        return np.mean(self.matrics[target_matric])

    def matric_var(self, target_matric):
        return np.var(self.matrics[target_matric])

    def matric_std(self, target_matric):
        return np.std(self.matrics[target_matric])

    def print_matric(self, detail_output=False):
        for target_matric in self.matrics_name:
            print(f"{target_matric} Len:{self.matric_len(target_matric)}, Ave:{self.matric_ave(target_matric)}, Std:{self.matric_std(target_matric)}")
            if detail_output:
                print(self.matrics[target_matric])

if __name__ == "__main__":
    file_path = sys.argv[1]
    log_reader = LogAnalyzer(file_path)
    log_reader.print_matric(False)
    
