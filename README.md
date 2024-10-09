# TourMix1M

## Dataset

TourMix1M is a large-scale multimodal dataset consisting of one million review instances related to Japanese tourist spots. It includes reviews under various conditions such as images, user attributes, user profiles, review ratings, review lengths, key phrases, and visit seasons.

### Data Example

Here's an example of data included in the dataset:

![Data Example](images/data_example.png)

### Dataset Statistics

The main statistics of the dataset are as follows:

| Component | Count |
|-----------|-------|
| Dialogues | 1,000,000 |
| Prompts | 1,310,000 |
| Reviews | 545,891 |
| Images | 476,167 |
| Tourism Spots | 51,011 |

![Dataset Statistics Graph](images/dataset_statistics.png)

## Model

LLaVA-Review is a large-scale multimodal model fine-tuned on the TourMix1M dataset. It takes images and natural language instructions as input and generates tourism reviews.

### Model Architecture

![LLaVA-Review Model Architecture](images/model_architecture.png)

## Results of General Review Generation

### Performance Comparison Table

| Model | BLUE | ROUGE-1 | ROUGE-L | CIDEr | DIV | PROPN | TFIDF-F1 | Senti-F1 | length |
|-------|------|---------|---------|-------|-----|-------|----------|----------|--------|
| LLaVA-1.5 | 0.683 | 0.254 | 0.162 | 0.099 | 0.863 | 0.303 | 0.141 | 0.029 | 133.8 |
| ChatGPT-4V | 0.622 | 0.250 | 0.165 | 0.103 | 0.955 | 0.278 | 0.169 | 0.036 | 70.9 |
| LLaVA-Review | 1.291 | 0.272 | 0.185 | 0.108 | 0.920 | 0.457 | 0.244 | 0.045 | 54.9 |

### Generation Example

![General Review Generation Example](images/general_review_generation_example.png)

## Results of Conditional Review Generation

### Performance Comparison Table

| Model | BLUE | ROUGE-1 | ROUGE-L | CIDEr | DIV | PROPN | TFIDF-F1 | Senti-F1 | length |
|-------|------|---------|---------|-------|-----|-------|----------|----------|--------|
| LLaVA-Review | 1.291 | 0.272 | 0.185 | 0.108 | 0.920 | 0.457 | 0.244 | 0.045 | 54.9 |
| LLaVA-Review + gender | 1.410 | 0.269 | 0.185 | 0.106 | 0.920 | 0.430 | 0.239 | 0.046 | 54.3 |
| LLaVA-Review + length | 1.952 | 0.308 | 0.198 | 0.184 | 0.923 | 0.510 | 0.244 | 0.048 | 87.3 |

### Generation Examples

1. Conditioning on User Attributes

![User Attribute Conditioning Example](images/conditional_generation_user_attributes.png)

2. Conditioning on Review Length

![Review Length Conditioning Example](images/conditional_generation_review_length.png)

3. Conditioning on Key Phrases

![Key Phrase Conditioning Example](images/conditional_generation_key_phrase.png)
