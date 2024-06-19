# Comparative Performance Analysis of LLMs with ML models on Sentiment Analysis
The above code files were used for the implementation of ML models, LLMs for sequence classification task. 
Instead of using traditional techniques such TF-IDF, Word2Vec, etc. I utilized the BERT model to generate embeddings as inputs for ML models.

# LLMs used

- Llama 2-7B
- Falcon-7B
- Mistral-7B
- Zephyr-7B

# ML models used

- SVM
- Logistic Regression
- Naive-Bayes
- Random Forest
- Gradient Boosting

# Datasets

- Tweets - 100,000
- App Reviews - 50,000
- Yelp Reviews - 30,000

# Methodology

The models were initially fine-tuned using the tweets dataset, followed by further fine-tuning on the App Reviews dataset. Their performance was then evaluated using the Yelp Reviews dataset.

Among all the models, Falcon-7B showed superior performance.

# Requirements

Due to the size of the model adapters and datasets, executing this code was only feasible on a High Performance Computing Cluster (HPCC).
