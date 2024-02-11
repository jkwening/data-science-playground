# [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/overview)

Can you predict how capable each applicant is of repaying a loan?

## Description

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

**Home Credit** strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## Evaluation

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

Submission File
For each SK_ID_CURR in the test set, you must predict a probability for the TARGET variable. The file should contain a header and have the following format:

```text
SK_ID_CURR,TARGET
100001,0.1
100005,0.9
100013,0.2
etc.
```

@misc{home-credit-default-risk,
    author = {Anna Montoya, inversion, KirillOdintsov, Martin Kotek},
    title = {Home Credit Default Risk},
    publisher = {Kaggle},
    year = {2018},
    url = {<https://kaggle.com/competitions/home-credit-default-risk}>
}
