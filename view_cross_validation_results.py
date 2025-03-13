import pandas as pd
import os
import argparse
from typing import List
import numpy as np
import json
import re

def file_search(src_path: str, tgt_file: str):

    def recurse(tgt_path: str, tgt_file: str, output_list: List):

        if os.path.isdir(tgt_path):
            for file in os.listdir(tgt_path):
                recurse(os.path.join(tgt_path, file), tgt_file, output_list)
        else:
            if os.path.basename(tgt_path) == tgt_file:
                output_list.append(tgt_path)

    output_list = list()

    recurse(src_path, tgt_file, output_list)
    
    return output_list

def main():

    parser = argparse.ArgumentParser(
        'view_raytune_results',
    )

    parser.add_argument(
        'source',
    )

    src_directory = parser.parse_args().source
    target_file = 'result.json'

    paths = file_search(src_directory, target_file)

    results = pd.DataFrame()

    for file in paths:
        with open(file, 'r') as f:
            data = json.load(f)

            result = pd.DataFrame([data['config']])

            avg_loss = data['avg_loss']

            match = re.search(r"tf\.Tensor\(([-+]?\d*\.\d+|\d+)", data["avg_acc"])
            avg_acc = float(match.group(1)) if match else None

            result["avg_loss"] = avg_loss
            result["avg_acc"] = avg_acc
        
            results = pd.concat([results, result], ignore_index=True)
    
    print(results)
    results.to_csv('results.csv')

if __name__ == "__main__":
    main()