# Streamlit Cloud Deployment Guide

## Prerequisites

- Python 3.10 configured via `runtime.txt`
- All dependencies pinned in `requirements.txt`
- Repository pushed to GitHub

## Deployment Steps

### 1. Prepare Your Repository

Make sure you have committed all changes to Git:
```bash
git add .
git commit -m "Update deployment configuration for Python 3.10"
git push
```

### 2. Create `.streamlit/secrets.toml` on Streamlit Cloud

- Go to [Streamlit Cloud](https://share.streamlit.io)
- Click "New app"
- Select your repository and branch
- In Advanced Settings, add your secrets:
  ```
  GROK_API_KEY = "your_actual_grok_api_key"
  ```

### 3. Set Up Secrets Locally (for testing)

Create `.streamlit/secrets.toml` locally (this is in .gitignore):
```toml
GROK_API_KEY = "your_test_api_key"
```

**IMPORTANT:** Never commit this file to Git!

### 4. Test Locally

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

## Environment Variable Handling

The app automatically detects the deployment environment:

### Local Development
- Loads from `.env` file (via `python-dotenv`)
- Falls back to environment variables

### Streamlit Cloud
- Loads from Streamlit secrets (`.streamlit/secrets.toml`)
- No `.env` file is committed to Git

## Troubleshooting

### TensorFlow ImportError
- Ensure `runtime.txt` contains `python-3.10.13`
- Verify `requirements.txt` has `tensorflow==2.15.0`

### Missing API Keys
- Check `.streamlit/secrets.toml` on Streamlit Cloud dashboard
- Verify key names match: `GROK_API_KEY`

### Model Loading Error
- Ensure `model/animal_model.h5` is in the repository
- Check that the file path is correct in main.py

## Project Structure

```
animal-classification/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies (pinned versions)
├── runtime.txt             # Python version (3.10.13)
├── .gitignore             # Excludes .env and secrets.toml
├── .streamlit/
│   ├── secrets.toml        # (NOT in Git) - Local secrets only
│   └── secrets.toml.example # Template for secrets
├── model/
│   ├── animal_model.h5    # Pre-trained model
│   └── classes.json       # Class labels
├── animals/               # Dataset folder
├── src/
│   ├── predict.py         # Prediction script
│   ├── train.py          # Training script
│   └── utils.py          # Utilities
└── README.md             # Project documentation
```

## Key Changes for Deployment

1. **Python Version**: Specified in `runtime.txt` (3.10.13)
2. **Dependencies**: All pinned to compatible versions
3. **Secrets Management**: Uses `st.secrets` for cloud, `.env` for local
4. **API Keys**: Never committed to Git (excluded in .gitignore)