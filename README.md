# Health Insurance Analyzer

A Streamlit web app that analyses Indian health insurance policy documents and lets users chat with an AI underwriter about their specific policy. Powered by Claude Sonnet 4.5.

## Features

- **Upload** a health insurance policy as PDF or image (PNG / JPG / WEBP)
- **Automatic validation** — rejects non-Indian policies and documents missing terms & conditions
- **Structured extraction** of 12 key parameters (Sum Insured, Room rent, ICU, Pre / Post hospitalisation, Co-pay, Day care, OPD, PED, Maternity, Permanent exclusions, Specific illness waiting period)
- **Chat** with an AI underwriter grounded strictly on your uploaded policy (150-word answers, session-scoped memory)
- **Hybrid PDF handling** — fast text extraction for digital PDFs, vision fallback for scanned ones
- **Privacy first** — API key and policy data are kept in the browser session only; nothing is persisted server-side

## Local setup

```bash
git clone <your-repo-url>
cd health-insurance-analyzer
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, paste your Anthropic API key in the sidebar, and go.

## Deploying to Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app**, select the repo and branch, and set the main file path to `app.py`.
4. Click **Deploy**. No secrets are needed — users enter their own Anthropic API key in the app's sidebar.

Deployment typically takes 2–3 minutes.

## Usage

1. **Enter your Anthropic API key** in the sidebar (get one at [console.anthropic.com](https://console.anthropic.com)).
2. **Upload** your health insurance policy (PDF or image).
3. **Tick the disclaimer** checkbox.
4. Click **Analyse policy**.
5. Review the 12-parameter table, then ask follow-up questions in the chat box.
6. Use the **Reset session** button in the sidebar to start over with a new policy.

## Error messages you might see

| Scenario | Message |
|---|---|
| Non-Indian health insurance (or non-health) document | `Incorrect document uploaded, please upload the correct document` |
| Indian policy but no terms & conditions (e.g., only a policy schedule) | `Incomplete document uploaded without the terms and conditions, please upload the correct document` |
| Invalid API key | `Invalid API key. Please check the key in the sidebar and try again.` |

## Configuration

The model is pinned to `claude-sonnet-4-5-20250929` in `app.py`. To upgrade, change the `MODEL` constant at the top of the file (for example, to `claude-sonnet-4-6`).

## Disclaimer

This tool provides an AI-generated analysis for **informational purposes only**. Always cross-reference with the original policy document and / or a qualified insurance advisor before making any practical decisions.
