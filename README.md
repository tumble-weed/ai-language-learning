# AI Language Learning

🎙️ **Language Learning Difficulty Analyzer** - Analyze audio recordings to identify sentence difficulty based on readability metrics.

## 🔐 Authentication

This app uses **Google OAuth 2.0** for secure user authentication. Users must sign in with their Google account to access the application features.

### ✨ Persistent Login
Once logged in, you'll **stay logged in** even after closing your browser! The app uses encrypted cookies to remember your authentication.

### Quick Setup

1. **Get Google OAuth Credentials**:
   - See [GOOGLE_AUTH_SETUP.md](GOOGLE_AUTH_SETUP.md) for detailed instructions
   - Create a project in [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Google+ API
   - Create OAuth 2.0 credentials

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Google credentials + generate a COOKIE_PASSWORD
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

For detailed authentication flow, see [AUTH_FLOW.md](AUTH_FLOW.md).  
For persistent authentication details, see [PERSISTENT_AUTH.md](PERSISTENT_AUTH.md).

## Download IndicLID model and save it in indiclid-ftn folder
` https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.zip `


```mermaid
flowchart TD

A([Start Program]) --> B[Load CSV File]
B --> C[Load or Create User Profile]

C --> D[Initialize Glicko2 & FSRS Scheduler]
D --> E[Show Current Skill Level]

E --> F[Select Sentence]
F --> F1{Any FSRS Due\nwithin Skill Window?}

F1 -->|Yes| F2[Pick Due Sentence]
F1 -->|No| F3[Sample New Sentence\nby Difficulty]

F2 --> G[Show Sentence to User]
F3 --> G

G --> H[User Inputs Answer]
H --> I{User Quit?}

I -->|Yes| J[Save User Data]
J --> K([Exit Program])

I -->|No| L[Evaluate Answer]

L --> M[Show Correct/Incorrect]

M --> N[Glicko Rating Update]
N --> N1[Update Rating, RD, Sigma]
N1 --> N2[Convert to 1–10 Skill]

N2 --> O[FSRS Review Update]
O --> O1{Card Exists?}

O1 -->|No| O2[Create New Card]
O1 -->|Yes| O3[Use Existing Card]

O2 --> O4[Apply FSRS Review]
O3 --> O4

O4 --> O5[Update Next Due Date]

O5 --> P[Show New Skill & RD]
P --> E
