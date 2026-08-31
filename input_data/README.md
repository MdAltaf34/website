# Optional repository input data

You do not need to put data here if you use the **Upload one input ZIP** option in the website.

If you prefer to keep the demo inputs in the GitHub repository, use this layout:

```text
input_data/
├── daily_sentiment_banglabert.csv
├── daily_sentiment_lexicon.csv
└── dse-unadjusted-data/
    ├── 00DSEX_data.csv
    ├── 00DS30_data.csv
    ├── 00DSES_data.csv
    ├── GP_data.csv
    ├── ACI_data.csv
    ├── BEXIMCO_data.csv
    ├── BRACBANK_data.csv
    └── BXPHARMA_data.csv
```

The sentiment CSV files are the already-generated daily sentiment scores. This demo does not rerun article-level BanglaBERT inference.
