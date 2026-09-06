# llm-content-moderation-benchmark
## Dataset Setup
1. The raw dataset `train.csv` used to generate the sampled dataset used in this project can be downloaded from the [Kaggle Jigsaw Competition](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv)
2. The raw dataset should then be placed inside `data/`
3. Running `python src/data_loader.py` should reproduce the sampled benchmark dataset
