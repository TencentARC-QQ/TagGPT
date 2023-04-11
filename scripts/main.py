import os
import pandas as pd
import tqdm
import numpy as np
import argparse

from langchain import OpenAI, ConversationChain
from langchain.prompts import PromptTemplate
from collections import Counter
from sentence_transformers import SentenceTransformer


class Data:
    def __init__(self, path):
        self.path = path
        self.dataframe = self.data_loader()

    def data_loader(self):
        df = pd.read_csv(self.path, sep=';', usecols=['caption', 'ocr_cover', 'asr_pure', 'category_name'],
                         on_bad_lines='skip')

        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="data path")
    parser.add_argument("--func", type=str, help="func")
    parser.add_argument("--gen_feq", type=int, help="gen_feq")

    paras = parser.parse_args()

    data_path = paras.data_path
    func = paras.func
    gen_feq = paras.gen_feq




if __name__ == "__main__":
    main()