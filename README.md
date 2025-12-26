# AI Language Learning

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
