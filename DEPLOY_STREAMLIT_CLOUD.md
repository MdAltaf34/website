# Deploy to Streamlit Community Cloud

## 1. Upload the project to GitHub

Extract the deployment ZIP and upload the **contents** to a GitHub repository. `app.py` and `requirements.txt` must be at the repository root.

## 2. Create the Streamlit app

In Streamlit Community Cloud:

- Repository: your GitHub repository
- Branch: `main`
- Main file path: `app.py`
- Python: **3.12**

## 3. Configure the administrator password

In Streamlit App settings -> Secrets, add:

```toml
ADMIN_PASSWORD = "your-private-password"
```

Do not put the real password in GitHub.

## 4. Open the pages

For an app named `dse-forecast`:

- Public: `https://dse-forecast.streamlit.app/`
- Administrator: `https://dse-forecast.streamlit.app/admin`

The administrator page is hidden from navigation but remains password-protected.

## 5. First computation

On `/admin`:

1. Log in.
2. Upload the input ZIP.
3. Validate the ZIP.
4. Run cloud computation and publish the February forecast.
5. Download `app_data.zip` after the run completes.

## 6. Save the first result permanently

Because Streamlit's runtime storage can reset, extract the downloaded `app_data.zip` and commit the resulting files to:

`data/app_data/`

Push the commit to GitHub. Streamlit will redeploy and thereafter the public page can display the saved forecast immediately without a new upload or model run.

Use `/admin` again only when you intentionally want to recompute and replace the published result.
