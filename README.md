# TourMix1M

## License
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](http://creativecommons.org/licenses/by-nc-sa/3.0/).　This dataset is intended for research purposes only and as such cannot be used commercially

## Dataset
TourMix1M is a large-scale multimodal dataset consisting of one million review instances related to Japanese tourist spots. It includes reviews under various conditions such as images, user attributes, user profiles, review ratings, review lengths, key phrases, and visit seasons.

### Dataset Statistics

The main statistics of the dataset are as follows:

| Component | Count |
|-----------|-------|
| Dialogues | 1,000,000 |
| Prompts | 1,310,000 |
| Reviews | 545,891 |
| Images | 476,167 |
| Tourism Spots | 51,011 |

![Dataset Statistics Graph](readme_images/task_pie_wide.png)

## Model

LLaVA-Review is a large-scale multimodal model fine-tuned on the TourMix1M dataset. It takes images and natural language instructions as input and generates tourism reviews.

### Model Architecture

![LLaVA-Review Model Architecture](readme_images/llavareview_arch.png)

## Results of General Review Generation

### Performance Comparison Table

| Model | BLUE | ROUGE-1 | ROUGE-L | CIDEr | DIV | PROPN | TFIDF-F1 | Senti-F1 | length |
|-------|------|---------|---------|-------|-----|-------|----------|----------|--------|
| LLaVA-1.5 | 0.683 | 0.254 | 0.162 | 0.099 | 0.863 | 0.303 | 0.141 | 0.029 | 133.8 |
| ChatGPT-4V | 0.622 | 0.250 | 0.165 | 0.103 | 0.955 | 0.278 | 0.169 | 0.036 | 70.9 |
| LLaVA-Review | 1.291 | 0.272 | 0.185 | 0.108 | 0.920 | 0.457 | 0.244 | 0.045 | 54.9 |

### Generation Example

![General Review Generation Example](readme_images/general_rg_example.png)

## Results of Conditional Review Generation

### Generation Examples


![User Attribute Conditioning Example](readme_images/cond_rg_examples.png)

