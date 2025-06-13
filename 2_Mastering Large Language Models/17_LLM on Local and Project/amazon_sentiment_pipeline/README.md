Amazon Review Dataset of electronics (3 million values)

which contains text reviews of products what we can classifiy into 3 sentiments:
1. negative (1-2)
2. neutral (3)
3. positive (4-5)

model - llama
for sentiment classification
- zero-shot
- few-shot

pipeline
- process reviews
- predict sentiment
- evaluate performance
- visualize results

steps

1. data_preprocessing.py - loading the data and cleaning it
2. sentiment_classifier.py - sentiment classification which is Llama based
3. evaluation.py - model evaluation and metrics
4. visualization.py - visualize the output
5. pipeline.py - (main) orchestration for all the python files and project


## dataset from kaggle API (terminal)
pip install kaggle

kaggle datasets download -d cynthiarempel/amazon-us-customer-reviews-dataset -p data/raw unzip data/raw/amazon_reviews_us_Electronics_v1_00.tsv.zip -d data/raw

directly download it here: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?resource=download&select=amazon_reviews_us_Electronics_v1_00.tsv

