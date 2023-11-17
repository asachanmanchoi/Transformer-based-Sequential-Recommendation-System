# Transformer-based Sequential Recommendation

This is the research project `"Exploring Temporal Factors in Transformer-based Recommendation Systems for Sequential Recommendation"` for COMP 5331, which focuses on Knowledge Discovery in Databases.

## Inspiration

Our work and code is inspired by the KDD 2023 paper [Text Is All You Need: Learning Language Representations for Sequential Recommendation](https://arxiv.org/abs/2305.13731) and [Recformer](https://github.com/JiachengLi1995/Recformer) implementation.


## Dependencies

- Python 3.10.10
- PyTorch 2.0.0
- PyTorch Lightning 2.0.0
- Transformers 4.28.0
- Deepspeed 0.9.0

## Pretrained Model
Download the pretrained model from the following links and put them in the `./pretrained_model` folder.
|              Model              |
|:-------------------------------|
|[RecformerModel](https://drive.google.com/file/d/1aWsPLLgBaO51mPqzZrNdPmlBkMEZ-naR/view?usp=sharing)|
|[RecformerForSeqRec](https://drive.google.com/file/d/1BEboY3NxAUOBe6YwYZ_RsQ4BR6IIbl0-/view?usp=sharing)|

## Zero-shot
### Dataset
We use several categories in [Amazon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) to evaluate the zero-shot recommendation performance.

You can process the raw data using our provided scripts `process_data/process.py`. You need to set meta data path `--meta_file_path`, interaction data path `--file_path`, output path `--output_path`, and `--sample_rate` (the default value is 1) in `process_data/process.sh`, and execute the script:
```bash
cd process_data
bash process.sh
```
### Evaluation
You can evaluate the zero-shot recommendation performance using our provided scripts `zero_shot.py`. You need to set the model path `--model_ckpt` and the dataset path `--dataset_path` in `zero_shot.sh` and execute the script:
```bash
bash zero_shot.sh
```
Our code will evaluate the zero-shot recommendation performance of the given model on the dataset and return the following metrics:
- NDGC@10, NDGC@50
- Recall@10, Recall@50
- MRR
- AUC

## Finetuning
### Dataset
We use several categories in [Amazon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) to finetune our model.

You can process the raw data using our provided scripts `process_data/process.py`. You need to set meta data path `--meta_file_path`, interaction data path `--file_path`, output path `--output_path`, and `--sample_rate` (the default value is 1) in `process_data/process.sh`, and execute the script:
```bash
cd process_data
bash process.sh
```
### Training
You can finetune our model using our provided scripts `finetune.py`. In `finetune.sh`, you need to set the model path `--pretrain_ckpt`, the dataset path `--data_path`, and you can also set the maximum nunber of training epoch `--num_train_epochs`. To load the best item embedding table from stage 1 before getting into stage 2 of training, set `--reload_item_embeddings` to `True`, otherwise set it to `False`. You can execute the script by:
```bash
bash finetune.sh
```
Our code will finetune the pretrained model on the dataset and save the finetuned model and the evaluation results in the `./checkpoints` folder.

<!-- ## Assistance for members

1. [How to use Git](members/tutorial/Git.md) -->

## Milestone

| Date                       | Weekday | Group Meeting                                | Update                                                              | Note |
|----------------------------|---------|----------------------------------------------|---------------------------------------------------------------------|------|
| 2023-09-11 (20:30 - 22:00) | Monday  | 1 (Project Content Discussion Meeting)       | Determine to focus on "Recommendation System"                       |      |
| 2023-09-16 (20:30 - 22:30) | Saturday| 2 (Read 13 Related Papers Meeting)           | Determine to read only two papers after discussion                  |      |
| 2023-09-24 (14:00 - 16:00) | Sunday  | 3 (Project Proposal Discussion Meeting)      | Discuss the project proposal content                                |      |
| 2023-09-26 (21:00 - 22:00) | Tuesday | 4 (Project Proposal Review Meeting)          | Review the project proposal                                         |      |
| 2023-10-15 (15:30 - 17:30) | Sunday  | 5 (Dataset preprocessing Discussion Meeting) | Determine how to preprocess the dataset and plan in the next week   | [click](./meeting/fifth/README.md) |
| 2023-10-22 (21:00 - 22:20) | Monday  | 6 (Source Code Running Insight Meeting)      | Share insights during the phase of running source code              | [click](./meeting/sixth/README.md) |


## Contributors & Maintainers

[@Elaine, Xiaohan ZHONG](https://github.com/ElaineXHZhong)
[@TAN, Weile](https://github.com/Ust-Waylon)
[@Ng Ting Kwong]()
[@WANG, Liangda]()
[@CHAN, Man Choi]()
[@DAI, Ruyi]()

## Contribute

Contributions are always welcome! Feel free to dive in! 

Please read the [contribution guideline](https://github.com/github/docs/blob/main/CONTRIBUTING.md) first, then [open an issue](https://github.com/ElaineXHZhong/Content-Sentiment-Analysis/issues/new) open an issue</a> or submit PRs.

This repository follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

## License

[MIT](LICENSE) © Elaine, Xiaohan ZHONG
