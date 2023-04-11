# TagGPT
TagGPT is a fully automated system capable of tag extraction and multimodal tagging in a completely zero-shot fashion.


Paper Link: [TagGPT: Large Language Models are Zero-shot Multimodal Taggers](https://arxiv.org/abs/2304.03022)

<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" href="https://huggingface.co/spaces/TencentARC/TagGPT">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" alt="Open in Spaces">
</a>

## Dependencies

- Python >= 3.7
- PyTorch == 2.0.0
- transformers==4.27.4

```bash
pip install -r requirements.txt
```

## How to use TagGPT

### Step 1: Tagging system construction
You need a batch of data to build your tagging system.
Here, we can use the Kuaishou open source data, which you can download [here](https://pan.baidu.com/s/1v6x14o5K9IuM3A-IS29UoA?pwd=ihc2#list/path=%2F) (password: ihc2).

First, you can place the data in the './data/' folder and format it with the following command.
```bash
python ./scripts/main.py --data_path ./data/222k_kw.ft --func data_format
```

Then, you can use the following command to generate candidate tags based on LLMs.
```bash
python ./scripts/main.py --data_path ./data/sentences.txt --func tag_gen --openai_key "put your own key here" --gen_feq 5
```

Next, the tagging system can be obtained by post-processing.
```bash
python ./scripts/main.py --data_path ./data/tag_gen.txt --func posterior_process
```

### Step 2: Data tagging
TagGPT can assign tags to the given samples based on the built tagging system, and you can adapt your data to what './data/examples.csv looks like.

And TagGPT provides two different tagging paradigms:
1. Generative tagger

```bash
python main.py --data_path ../data/examples.csv --tag_path ../data/final_tags.csv --func selective_tagger --openai_key "put your own key here"
```
2. Selective tagger

```bash
python main.py --data_path ../data/examples.csv --tag_path ../data/final_tags.csv --func generative_tagger --openai_key "put your own key here"
```