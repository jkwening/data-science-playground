# Overview
The purpose of this repository is to house mini-projects that serve my on-going education in data-science skills. This mini-projects will be in varying states of completion. The primary goal of this is to try knew things and build upon skills and knowledge I learn.

Below is a brief summary of what each folder contains and goal/intention behind spending time on them.

## search-engine
Apply knowledge of Term Frequency - Inverse Document Frequency (TF-IDF) to implement simple search engine using a datasets of 600 abd 25,000+ Wikipedia articles. The objectives of this porject - (a) understand and apply TF-IDF on a realistic task, (b) see what solving an NLP problem looks like end-to-end and (c) understand the fundamentals of how a search engine works.

Overview
The search ranking function is going to be quite simple. Suppose the query is "albert einstein". Then, score(article) = TF-IDF(article, "albert") + TF-IDF(article, "einstein"). The search engine will display the articles sorted by score.

Calculating the TF-IDF scores involves:
1. Compute term frequency and document frequency statistics
2. Use TF-IDF metrics to calculate the final score and rank the articles

Note: Datasets are too large to store in github; will eventually post a link for access. Feel to reach in the interim if you want them. Thanks!
