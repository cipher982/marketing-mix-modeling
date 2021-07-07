# marketing-mix-modeling

This repo contains two notebooks:
- A super simple baseline using sklearn and feature importance
- An in-depth Bayesian STAN model


The Bayseian STAN notebook is much more interesting and follows a modern attempt to model a marketing mix of spending and learn the most effective channels of spending based on spending, impressions, and sales. It takes into account seasonlity, market conditions, time-lag effects of marketing, and more.

Sources for implementation:
- Original paper: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf
- Code: https://github.com/sibylhe/mmm_stan
