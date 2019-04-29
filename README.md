# Overview
The purpose of this repository is to house mini-projects that serve my on-going education in data-science skills. This mini-projects will be in varying states of completion. The primary goal of this is to try knew things and build upon skills and knowledge I learn.

Below is a brief summary of what each folder contains and goal/intention behind spending time on them.

## search-engine
Applies knowledge of TF-IDF to implement simple search engine using a datasets of 600 abd 25,000+ Wikipedia articles. The objectives of this porject - (a) understand and apply TFIDF on a realistic task, (b) see what solving an NLP problem looks like end-to-end and (c) understand the fundamentals of how a search engine works.

Overview
The search ranking function is going to be quite simple. Suppose the query is "albert einstein". Then, score(article) = TFIDF(article, "albert") + TFIDF(article, "einstein"). The search engine will display the articles sorted by score.

Calculating the TF-IDF scores involves:
1. Compute term frequency and document frequency statistics
2. Use TF-IDF metrics to calculate the final score and rank the articles
