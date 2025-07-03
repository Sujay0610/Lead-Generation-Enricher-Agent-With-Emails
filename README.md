 # 🎯 Lead Generation System

This project implements an AI-powered lead generation system that leverages Google Custom Search, Apify for data enrichment, and Google Sheets for saving scraped lead data. It's built as a Streamlit application, allowing for interactive lead generation.

## ✨ Features

- **AI-Powered Lead Generation**: An intelligent agent (Lead Generation Joe) understands natural language queries to find leads.
- **Google Custom Search Integration**: Searches LinkedIn profiles based on specified criteria (location, business, job title).
- **Apify Data Enrichment**: Automatically enriches scraped LinkedIn profiles with detailed information like email, phone number, company details, experience, and more.
- **Google Sheets Integration**: Seamlessly saves all enriched lead data into a Google Sheet, avoiding duplicates.
- **Individual Lead Research**: Ability to research a single LinkedIn profile and save its details.
- **Interactive UI**: User-friendly interface built with Streamlit for easy interaction and progress tracking.

## ⚙️ Setup

To run this project, you will need to set up several API keys and Google Sheets credentials.

### 1. Environment Variables / Streamlit Secrets

Create a `.streamlit/secrets.toml` file in your project root with the following keys:

```toml
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
GOOGLE_CSE_ID = "YOUR_GOOGLE_CUSTOM_SEARCH_ENGINE_ID"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" # Used for the LLM agent
APIFY_API_TOKEN = "YOUR_APIFY_API_TOKEN"

# Google Sheets Service Account Credentials
# Replace with your actual service account key JSON content, formatted as a single string
GOOGLE_SHEETS_CREDENTIALS = """
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\nyour_private_key\n-----END PRIVATE KEY-----\n",
  "client_email": "your-service-account-email@your-project-id.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account-email.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
"""
```

**How to get these credentials:**

-   **GOOGLE_API_KEY & GOOGLE_CSE_ID**: Follow Google's guide to set up a Custom Search Engine and get an API key.
-   **OPENAI_API_KEY**: Obtain this from your OpenAI account.
-   **APIFY_API_TOKEN**: Get this from your Apify dashboard. You will be using the `dev_fusion~linkedin-profile-scraper` actor.
-   **GOOGLE_SHEETS_CREDENTIALS**: 
    1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
    2. Create a new project or select an existing one.
    3. Enable the "Google Drive API" and "Google Sheets API" for your project.
    4. Go to "APIs & Services" > "Credentials".
    5. Click "Create Credentials" > "Service Account".
    6. Fill in the details and grant "Project" > "Editor" role (or more restrictive if preferred).
    7. After creating, click on the service account email, then "Keys" tab, and "Add Key" > "Create new key" > "JSON". Download the JSON file.
    8. Open the downloaded JSON file and copy its entire content into the `GOOGLE_SHEETS_CREDENTIALS` variable in `secrets.toml`, ensuring it's a single-line string with escaped quotes if necessary (though triple quotes `"""` should handle it).
    9. **Important**: Share your target Google Sheet with the client email address from your service account JSON file (e.g., `your-service-account-email@your-project-id.iam.gserviceaccount.com`).

### 2. Install Dependencies

First, make sure you have Python installed (3.8+ recommended).

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

To start the Streamlit application, navigate to the project directory in your terminal and run:

```bash
streamlit run leadagent.py
```

This will open the application in your web browser.

## 📝 Usage

1.  **Start Chat**: Begin by saying "Hi" to Lead Generation Joe.
2.  **Provide Lead Criteria**: Joe will guide you to provide details like:
    -   `Locations`: e.g., "Chicago United States", "Sydney Australia"
    -   `Business`: e.g., "Financial Planners", "Software Development"
    -   `Job Titles`: e.g., "CEO", "Sales Manager"
3.  **Automated Process**: Once all information is gathered, Joe will:
    -   Search Google for LinkedIn profiles matching your criteria.
    -   Enrich each found profile using Apify.
    -   Save all enriched data to a Google Sheet named "PyLeads" (default, in "Sheet1").
4.  **Individual Research**: You can also ask Joe to "research a lead" by providing a specific LinkedIn URL.

## 📂 Project Structure

-   `leadagent.py`: The main Streamlit application script containing the AI agent logic, tool definitions, and UI.
-   `requirements.txt`: Lists all Python dependencies required for the project.
-   `.streamlit/secrets.toml`: (You need to create this) Stores your API keys and Google Sheets credentials securely.


