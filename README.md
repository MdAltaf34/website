# DSE One-Month Forecast — Final Streamlit Cloud Website

This package is deployment-ready for Streamlit Community Cloud.

## URLs after deployment

If the app URL is:

`https://dse-forecast.streamlit.app`

then:

- Public forecast page: `https://dse-forecast.streamlit.app/`
- Private administrator page: `https://dse-forecast.streamlit.app/admin`

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

## Permanent saved result after the first computation

Streamlit Community Cloud's runtime filesystem is not permanent. After the first successful computation:

1. On `/admin`, click **Download newly generated app_data.zip**.
2. Extract the downloaded ZIP.
3. Put the generated files in this repository under `data/app_data/`.
4. Commit and push those files to GitHub.

After Streamlit redeploys, the public website will load the saved results immediately. You do **not** need to upload inputs or rerun the models every time the app starts. The `/admin` page can remain available only when a future recomputation is needed.

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

## Streamlit Cloud deployment

1. Extract this ZIP.
2. Create a GitHub repository and upload the extracted project files so `app.py` is at the repository root.
3. Open Streamlit Community Cloud and create a new app from that repository.
4. Use branch `main` and main file `app.py`.
5. In Advanced settings, select **Python 3.12**.
6. Add this secret:

```toml
ADMIN_PASSWORD = "your-private-password"
```

7. Deploy.

Do not commit your real administrator password to GitHub.

## Local run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The public page is normally at `http://localhost:8501/` and the administrator page is at `http://localhost:8501/admin`.

## Research prototype notice

The website is a research prototype and the forecasts are not financial advice.
