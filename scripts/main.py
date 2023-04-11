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
from collections import Counter, defaultdict
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
        if tmp[1] > 2:
            tags.append(tmp[0].replace(' ', ''))

    f = open('../data/tags.txt', 'w')
    f.write('\n'.join(tags))
    f.close()

    encoder = SentenceTransformer('hfl/chinese-roberta-wwm-ext-large')
    tags_embed = encoder.encode(tags)
    tags_dis = [np.sqrt(np.dot(_, _.T)) for _ in tags_embed]
    mark = [0 for _ in range(len(tags))]
    include = [[] for _ in range(len(tags))]

    for i in tqdm.trange(len(tags)):
        if mark[i] == 0:
            score = np.dot(tags_embed[i], tags_embed[i:].T)
            for j in range(i, len(tags)):
                if i != j:
                    score[j - i] = score[j - i] / (tags_dis[i] * tags_dis[j])
                    if score[j - i] > 0.95:
                        mark[j] = 1
                        include[i].append(tags[j])

    out = ""
    for i in range(len(tags)):
        if mark[i] == 0:
            out += tags[i] + "||" + str(include[i]) + "\n"

    f = open('../data/final_tags.csv', 'w')
    f.write(out)
    f.close()


def get_tag_embed(encoder, tags):
    tags_embed = encoder.encode(tags)
    tags_dis = [np.sqrt(np.dot(_, _.T)) for _ in tags_embed]

    with open('../data/tags_embed.npy', 'wb') as f:
        np.save(f, tags_embed)

    with open('../data/tags_dis.npy', 'wb') as f:
        np.save(f, tags_dis)

    return tags_embed, tags_dis


def load_tag_embed():
    tags_embed = np.load('../data/tags_embed.npy')
    tags_dis = np.load('../data/tags_dis.npy')

    return tags_embed, tags_dis


def format_prompt_selective(data, candidate_tags):
    preference = "兴趣标签"

    examples_tags = [
        ['仓鼠迷宫', '仓鼠咬人', '抓老鼠', '虱子去除', '猫和老鼠', '猫咪寻找', '唇毛去除', '动物生活', '动物生育', '消灭老鼠',
         '笼子', '老鼠防治', '动物学', '动物吃播', '动物生产', '地笼捕鱼', '宠物笼子', '动物教学', '动物PK', '饵料雾化',
         '仓鼠生病', '笼子训练', '金丝熊笼子清洁', '笼子清洁', '动物', '动物秀', '仓鼠笼子', '透明笼子', '仓鼠运动',
         '去除小胡子', '仔猪拉稀', '灌汤小笼包', '动物生态', '消灭蟑螂'],
        ['勿忘我花束', '美腿秘籍', '腿型矫正', '学习简笔画', '绘画教学', '自学画画', '个人才艺', '瘦腿运动', '娱乐才艺', '快手才艺',
         '易学画法', '魔鬼训练营', '才艺大赏', '迷你世界', '才艺表达', '才艺选手', '简易画技巧', '纸艺才艺', '完美腿型', '瘦腿计划',
         '山地车速降', '才艺', '美腿锻炼', '漫画教学', '腿部健康', '动漫人物绘画教程', '降妖路亚竿', '高跟鞋简笔画教程', '腿型改善',
         '夜魔', '才艺项目', '腿型评估', '绘画技巧', '地球之极侣行', '创意才艺', '画裙子技巧', '美腿', '手魔人', '卫龙魔芋爽',
         '百里玄策'],
        ['压弯技巧', '剪枝技巧', '女性生活小技巧', '生活窍门', '生活技巧', '护指绷带', '小技巧', '生活保健', '钢筋绑扎技巧',
         '梨树修剪技巧', '生活调理', '断丝取出器', '指甲修剪技巧', '健康', '汽车小妙招', '夏季生活小技巧', '保健小妙招', '剪刀面',
         '钢丝绳插套', '健康生活', '生活小常识', '生活日常', '小康生活', '家居小技巧', '生理健康', '创意生活小技巧', '生活小妙招',
         '日常技巧', '健康养生', '学习小妙招', '正能量生活小妙招', '手机小妙招', '手动弯管器', '生活养生', '生活', 'DIY小妙招',
         '健身生活'],
        ['便宜的汽车', '汽车用品', '豪华轿车推荐', '豪华入门车', '汽车科二', '性价比最高汽车', '廉价跑车', '豪华车市场', '汽车精品',
         '入门级SUV', '国产豪华轿车', '豪华七座SUV', '汽车制造', '10万级SUV', '汽车电子', '汽车交车', '最便宜的面包车', '魏派汽车',
         '性价比高的汽车品牌', 'SUV汽车', '便宜的SUV', '经济实用轿车', '汽车DIY', '七座车推荐', '便宜好车', '性价比高的SUV', '汽车',
         '世界最贵的车', '性价比高的轿车推荐', '性价比高的跑车', '豪华SUV选购', '豪华座驾'],
        ['车内音响系统', '房产家居', '房屋户型', '音响调试', '汽车音响', '一层平房设计', '岁月静好', 'JBL音响', '智能音箱',
         '小户型厨房', '小户型家具', '路上风景', '小户型卫生间设计', '小户型装修', '北欧家居', '家居用品', '青春岁月',
         '小户型空间利用', '家居电气', '心路历程', '音响', '家居DIY', '100平米装修', '夜间行驶灯光操作', '车载音响安装', '小户型',
         '家居建材', '实用家居', '长途行车', '青春永驻', '追梦之路', 'BOSE音响', '时光流逝', '家居生活', '音响改装', '校园时光',
         '家居服', '居家', '音响配置', '三室两厅装修']
    ]
    examples = [
        "例如，给定一个视频，它的\"标题\"为\"笼子挺高的百分之八十不会跳出去，不知道是不是被什么吃掉了，但是也没有看见血，继续寻找biubiu，你们有"
        "没有找仓鼠的小办法\"，\"类别\"为\"动物\"，\"ocr\"为\"今天发现biubiu不见了,哪里都没有biubiu,昨天晚上笼子盖没有关\"，\"asr\"为\""
        "今天发现BB我不见了，哪里都没有BB昨天晚上笼子盖没有关，应该是跑出去了，但是这个笼子很高，一般跑不出去，加油找b区b区吧。\"，{}生成机器人"
        "从标签集合\"{}\"中推断出合理的\"{}\"为\"动物生活、仓鼠笼子、宠物笼子、仓鼠生病\"。"
        "".format(preference, '、'.join(examples_tags[0]), preference),
        "例如，给定一个视频，它的\"标题\"为\"不会画动漫腿？来看看你画的腿对不对 #动漫   #手绘教程   #手绘     #未来设计师\"，\"类别\"为\"才"
        "艺\"，\"ocr\"为\"不会画好看的漫画腿,跟我学画腿这样画更好看\"，\"asr\"为\"所以住万三另外的小路看我像吗，我每天都要做这么像漫步又会忘"
        "了身处妖精跳出物神秘的心情，要对全世界说，所以住万三另外在小路看我像吗？我和都要做这么像漫步又会忘了。\"，{}生成机器人从标签集合\"{}\""
        "中推断出合理的\"{}\"为\"学习简笔画、绘画教学、自学画画、简笔画技巧、绘画技巧、完美腿型、动漫人物绘画教程、漫画教学\"。"
        "".format(preference, '、'.join(examples_tags[1]), preference),
        "例如，给定一个视频，它的\"标题\"为\"日常生活小技巧 #生活小妙招  #内蒙特产\"，\"类别\"为\"健康,生活\"，\"ocr\"为\"生活小妙招招\"，"
        "\"asr\"为\"管的这些小技巧，知道你就捡到宝了，一插入吸管时容易弯折，只需用大拇指封住上端，就可以轻松他好二用剪刀给吸管，这样剪开剜着一下"
        "，就能把下水道的头发轻松取出来，三用小刀把西瓜呈螺旋状还看用它来收纳家里的电线电池太方便了四吸管剪去两头。留下中间的小弹簧，封口没喝完的"
        "酸奶很实用，关注我，了解更多生活小实验。\"，{}生成机器人从标签集合\"{}\"中推断出合理的\"{}\"为\"生活窍门、生活技巧、小技巧、生活小常"
        "识、家居小技巧、创意生活小技巧、生活小妙招、学习小妙招、DIY小妙招\"。"
        "".format(preference, '、'.join(examples_tags[2]), preference),
        "例如，给定一个视频，它的\"标题\"为\"长安系最便宜的轿车，4W起很多人都看不上它，但我知道车只是代步工具，又需要什么面子呢！ #长安汽车\"，"
        "\"类别\"为\"汽车\"，\"ocr\"为\"长安系最便宜的一款轿车\"，\"asr\"为\"我不否认现在的国产和合资还有一定的差距，但确实是他们让我们5万开"
        "了MP V8万开上了轿车，10万开张了ICV15万开张了大七座。\"，{}生成机器人从标签集合\"{}\"中推断出合理的\"{}\"为\"便宜的汽车、性价比最高"
        "的汽车、最便宜的面包车、性价比高的汽车品牌、经济实用轿车、便宜好车、性价比高的轿车推荐\"。"
        "".format(preference, '、'.join(examples_tags[3]), preference),
        "例如，给定一个视频，它的\"标题\"为\"全屋嵌入式低音音响，主要是这个投影仪真的是爱了💕 \"，\"类别\"为\"房产家居\"，\"ocr\"为\"42平,一"
        "室一厅小户型\"，\"asr\"为\"看，远方灯火闪亮着光。你一人低头在路上。这城市越大，越让人心慌多向往，多漫长。祝一路行李太多伤。把最初笑容都"
        "淡忘。时光让我们变得脆弱，却坚强，让我在爱青青对你唱。我多想能多陪你唱。把什么生的风景对你讲。\"，{}生成机器人从标签集合\"{}\"中推断出"
        "合理的\"{}\"为\"房屋户型、小户型家具、音响调试、小户型装修、小户型空间利用、小户型、家居生活\"。"
        "".format(preference, '、'.join(examples_tags[4]), preference)
    ]

    prompt = PromptTemplate(
        input_variables=["preference", "caption", "ocr", "asr", "category_name", "example", "candidate_tags"],
        template="你是一个视频的{preference}生成机器人，根据输入的视频标题、类别、ocr、asr从给定的标签集推理出合理的\"{preference}\"，"
                 "以多个多于两字的标签形式进行表达，以顿号隔开。{example}那么，给定一个新的视频，它的\"标题\"为\"{caption}\"，\"类别\"为"
                 "\"{category_name}\"，\"ocr\"为\"{ocr}\"，\"asr\"为\"{asr}\"，请从标签集合\"{candidate_tags"
                 "}\"中推断出该视频的\"{preference}\"："
    )

    example = examples[random.randint(0, 4)]
    text = prompt.format(preference=preference, caption=data['caption'],
                         category_name=data['category_name'], ocr_cover=data['ocr'],
                         asr_pure=data['asr'], example=example, candidate_tags="、".join(candidate_tags))

    return text


def selective_tagger(data_path, tag_path, api_key):
    openai.api_key = api_key

    df_exp = pd.read_csv(data_path, sep='\|\|', on_bad_lines='skip')
    df_tag = pd.read_csv(tag_path, sep='\|\|', on_bad_lines='skip')
    df_tag.columns = ['tag', 'contain_tags']
    tags = list(df_tag['tag'])

    encoder = SentenceTransformer('hfl/chinese-roberta-wwm-ext-large')
    if os.path.exists('../data/tags_dis.npy') and os.path.exists('../data/tags_embed.npy'):
        tags_embed, tags_dis = load_tag_embed()
    else:
        print("Generating tag embedding")
        tags_embed, tags_dis = get_tag_embed(encoder, tags)

    selective_tags = []

    for ind, row in enumerate(tqdm.tqdm(df_exp.iterrows())):
        inputs = [row[1]['caption'], row[1]['category_name'], row[1]['ocr'], row[1]['asr']]
        input_embed = encoder.encode(inputs)
        input_dis = [np.sqrt(np.dot(_, _.T)) for _ in input_embed]

        ans = np.dot(input_embed, tags_embed.T)
        for i in range(ans.shape[0]):
            for j in range(ans.shape[1]):
                ans[i][j] = ans[i][j] / (input_dis[i] * tags_dis[j])

        candidate_tags = []
        for i in range(ans.shape[0]):
            tmp = [_ for _ in zip(list(ans[i]), tags)]
            tmp = sorted(tmp, key=lambda x: x[0], reverse=True)
            candidate_tags += [_[1] for _ in tmp[:10]]

        candidate_tags = list(set(candidate_tags))
        text = format_prompt_selective(row[1], candidate_tags)

        final_res = []
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": text}],
                temperature=1.5,
                n=5
            )

            res = []
            for j in range(5):
                ans = completion.choices[j].message["content"].strip()
                ans = ans.replace("\n", "")
                ans = ans.replace("。", "")
                ans = ans.replace("，", "、")
                res += ans.split('、')

            final_res += res
            tag_count = defaultdict(int)
            for fr in final_res:
                if fr in candidate_tags:
                    tag_count[fr] += 1

            tag_count = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)

        except:
            tag_count = []
            print("api error")

        selective_tags.append(tag_count)

    return selective_tags


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
    parser.add_argument("--data_path", type=str, help="data path", default="")
    parser.add_argument("--tag_path", type=str, help="tag path", default="")
    parser.add_argument("--func", type=str, help="func", default="")
    parser.add_argument("--openai_key", type=str, help="openai key", default="")
    parser.add_argument("--gen_feq", type=int, help="gen_feq", default=5)

    paras = parser.parse_args()

    data_path = paras.data_path
    tag_path = paras.tag_path
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
        print("Posterior processing completed")
    elif func == "selective_tagger":
        results = selective_tagger(data_path, tag_path, openai_key)
        print("Tagging completed")
        print(results)





if __name__ == "__main__":
    main()