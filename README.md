# DSE One-Month Forecast — Final Streamlit Cloud Website

This package is deployment-ready for Streamlit Community Cloud.

## URLs after deployment



- Public forecast page: `https://dse-forecasting-site.streamlit.app/`
- Private administrator page: `https://dse-forecasting-site.streamlit.app/admin`

The administrator page is hidden from the public navigation and is also protected by `ADMIN_PASSWORD`.

## What the website does

The Administrator can upload one input ZIP, validate it, run the forecasting pipeline in Streamlit Cloud, and publish the generated forecast package.

The pipeline uses January 2022 for model/configuration selection and then retrains the selected combination using data available through January before producing the final blind February 2022 forecast.

The public page only reads the published files under `data/app_data/` and displays historical prices, the February forecast, the uncertainty interval, the selected model/configuration, and forecast metadata.

## Required administrator input ZIP

The uploaded ZIP should contain:

```text
input_bundle.zip
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

Ticker aliases DSEX/00DSEX, DS30/00DS30, and DSES/00DSES are supported by the pipeline.

Typical saved files are:

```text
data/app_data/
├── companies_summary.csv
├── run_manifest.json
├── DSEX_history.csv
├── DSEX_forecast.csv
├── DS30_history.csv
├── DS30_forecast.csv
├── DSES_history.csv
├── DSES_forecast.csv
├── GP_history.csv
├── GP_forecast.csv
├── ACI_history.csv
├── ACI_forecast.csv
├── BEXIMCO_history.csv
├── BEXIMCO_forecast.csv
├── BRACBANK_history.csv
├── BRACBANK_forecast.csv
├── BXPHARMA_history.csv
└── BXPHARMA_forecast.csv
```


```toml
ADMIN_PASSWORD = "123456"
```

## Local run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The public page is normally at `http://localhost:8501/` and the administrator page is at `http://localhost:8501/admin`.

## Research prototype notice

The website is a research prototype and the forecasts are not financial advice.
