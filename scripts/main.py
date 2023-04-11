import os
import pandas as pd
import tqdm
import numpy as np
import argparse
import random
import sys
import openai

from langchain import OpenAI, ConversationChain
from langchain.prompts import PromptTemplate
from collections import Counter
from sentence_transformers import SentenceTransformer


def format_data(data, preference):
    examples = [
        "例如，给定一个视频，它的\"标题\"为\"笼子挺高的百分之八十不会跳出去，不知道是不是被什么吃掉了，但是也没有看见血，继续寻找biubiu，你们有"
        "没有找仓鼠的小办法\"，\"类别\"为\"动物\"，\"ocr\"为\"今天发现biubiu不见了,哪里都没有biubiu,昨天晚上笼子盖没有关\"，\"asr\"为\""
        "今天发现BB我不见了，哪里都没有BB昨天晚上笼子盖没有关，应该是跑出去了，但是这个笼子很高，一般跑不出去，加油找b区b区吧。\"，{}生成机器人"
        "推断出合理的\"{}\"为\"仓鼠伪冬眠、仓鼠假死、仓鼠不见了、仓鼠冬眠\"。".format(preference, preference),
        "例如，给定一个视频，它的\"标题\"为\"不会画动漫腿？来看看你画的腿对不对 #动漫   #手绘教程   #手绘     #未来设计师\"，\"类别\"为\"才"
        "艺\"，\"ocr\"为\"不会画好看的漫画腿,跟我学画腿这样画更好看\"，\"asr\"为\"所以住万三另外的小路看我像吗，我每天都要做这么像漫步又会忘"
        "了身处妖精跳出物神秘的心情，要对全世界说，所以住万三另外在小路看我像吗？我和都要做这么像漫步又会忘了。\"，{}生成机器人推断出合理的\"{}"
        "\"为\"动漫老师、动漫人物绘画教程、漫画腿怎么画、绘画新手教程\"。".format(preference, preference),
        "例如，给定一个视频，它的\"标题\"为\"日常生活小技巧 #生活小妙招  #内蒙特产\"，\"类别\"为\"健康,生活\"，\"ocr\"为\"生活小妙招招\"，"
        "\"asr\"为\"管的这些小技巧，知道你就捡到宝了，一插入吸管时容易弯折，只需用大拇指封住上端，就可以轻松他好二用剪刀给吸管，这样剪开剜着一下"
        "，就能把下水道的头发轻松取出来，三用小刀把西瓜呈螺旋状还看用它来收纳家里的电线电池太方便了四吸管剪去两头。留下中间的小弹簧，封口没喝完的"
        "酸奶很实用，关注我，了解更多生活小实验。\"，{}生成机器人推断出合理的\"{}\"为\"日常生活小妙招、生活小技巧、小妙招大全\""
        "。".format(preference, preference),
        "例如，给定一个视频，它的\"标题\"为\"长安系最便宜的轿车，4W起很多人都看不上它，但我知道车只是代步工具，又需要什么面子呢！ #长安汽车\"，"
        "\"类别\"为\"汽车\"，\"ocr\"为\"长安系最便宜的一款轿车\"，\"asr\"为\"我不否认现在的国产和合资还有一定的差距，但确实是他们让我们5万开"
        "了MP V8万开上了轿车，10万开张了ICV15万开张了大七座。\"，{}生成机器人推断出合理的\"{}\"为\"长安轿车报价、最便宜的长安轿车、新款长安轿"
        "车\"。".format(preference, preference),
        "例如，给定一个视频，它的\"标题\"为\"全屋嵌入式低音音响，主要是这个投影仪真的是爱了💕 \"，\"类别\"为\"房产家居\"，\"ocr\"为\"42平,一"
        "室一厅小户型\"，\"asr\"为\"看，远方灯火闪亮着光。你一人低头在路上。这城市越大，越让人心慌多向往，多漫长。祝一路行李太多伤。把最初笑容都"
        "淡忘。时光让我们变得脆弱，却坚强，让我在爱青青对你唱。我多想能多陪你唱。把什么生的风景对你讲。\"，{}生成机器人推断出合理的\"{}\"为\"小"
        "户型装修、一室一厅装修、装修效果图\"。".format(preference, preference)
    ]
    sentences = []
    prompt = PromptTemplate(
        input_variables=["preference", "caption", "ocr_cover", "asr_pure", "category_name", "example"],
        template="你是一个视频的{preference}生成机器人，根据输入的视频标题、类别、ocr、asr推理出合理的\"{preference}\"，以多个多"
                 "于两字的标签形式进行表达，以顿号隔开。{example}那么，给定一个新的视频，它的\"标题\"为\"{caption}\"，\"类别\"为"
                 "\"{category_name}\"，\"ocr\"为\"{ocr_cover}\"，\"asr\"为\"{asr_pure}\"，请推断出该视频的\"{preference}"
                 "\"："
    )
    for ind, row in enumerate(tqdm.tqdm(data.iterrows())):
        example = examples[random.randint(0, 4)]
        caption = row[1]['caption'][:100]
        ocr_cover = row[1]['ocr_cover'][:100]
        asr_pure = row[1]['asr_pure'][:100]
        text = prompt.format(
            preference=preference,
            caption=caption,
            category_name=row[1]['category_name'],
            ocr_cover=ocr_cover,
            asr_pure=asr_pure, example=example
        )

        sentences.append(text)

    f = open('../data/sentences.txt', 'w')
    f.write("\n".join(sentences))
    f.close()


def tag_gen(data_path, openai_key, gen_feq):
    openai.api_key = openai_key

    sentences = []
    f = open(data_path, 'r')
    for line in f.readlines():
        sentences.append(line.strip())
    f.close()

    num = 0
    final_res = []
    for sentence in tqdm.tqdm(sentences):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": sentence}],
                temperature=1.5,
                n=gen_feq
            )

            res = str(num) + "||"
            for j in range(gen_feq):
                ans = completion.choices[j].message["content"].strip()
                ans = ans.replace("\n", "")
                res += str(ans) + "||"

            final_res.append(res)
        except:
            continue

        num += 1
        if len(final_res) == 100:
            f = open("../data/tag_gen.txt", 'a')
            f.write("\n".join(final_res))
            f.close()
            final_res = []


def posterior_process(data_path):
    f = open(data_path, 'r')
    out = ""
    tag_all = []
    for line in f.readlines():
        line = line.replace(".", "")
        line = line.replace("。", "")
        line = line.replace(",", "、")
        line = line.replace("，", "、")
        line = line.replace("'", "")
        line = line.replace("\n", "")
        line = line.replace("\"", "")
        tmp = line.strip().split('||')
        out += str(tmp) + "\n"
        for t in tmp:
            if '、' in t:
                tags = t.split('、')
                tag_all += tags
    f.close()

    ans = Counter(tag_all)
    ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)

    tags = []
    for tmp in ans:
        if tmp[1] > 4:
            tags.append(tmp[0].replace(' ', ''))

    f = open('../data/tags.txt', 'w')
    f.write('\n'.join(tags))
    f.close()


class Data:
    def __init__(self, path):
        self.path = path
        self.dataframe = self.data_loader()

    def data_loader(self):
        df = pd.read_feather(self.path)
        df_f = df[['item_id', 'caption', 'ocr_cover', 'asr_pure', 'category_name']]

        return df_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="data path")
    parser.add_argument("--func", type=str, help="func")
    parser.add_argument("--openai_key", type=str, help="openai key")
    parser.add_argument("--gen_feq", type=int, help="gen_feq", default=5)

    paras = parser.parse_args()

    data_path = paras.data_path
    func = paras.func
    gen_feq = paras.gen_feq
    openai_key = paras.openai_key

    if func == "data_format":
        format_data(data=Data(path=data_path).dataframe, preference="兴趣标签")
        print("Data formatting completed")
    elif func == "tag_gen":
        tag_gen(data_path, openai_key, gen_feq)
        print("Tag generation completed")
    elif func == "posterior_process":
        posterior_process(data_path)
        print("Posterior process completed")




if __name__ == "__main__":
    main()