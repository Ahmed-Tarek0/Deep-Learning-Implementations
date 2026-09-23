# HR Candidate Profile Parser

Parses a resume (PDF or plain text) into a clean, structured JSON object using
[LangChain](https://python.langchain.com/) and Google Gemini.

## What it does

Given a resume, the notebook extracts:

- `full_name`
- `email`
- `education` — list of `{degree, institution, year}`
- `skills` — list of strings
- `experience` — list of `{role, company, years}`

The output schema is defined with **Pydantic models** and enforced using LangChain's
`PydanticOutputParser`, so the result is always validated and type-checked (e.g. `year` is
guaranteed to be an integer, missing optional fields become `null` instead of breaking the
pipeline).

## How it works

1. **Input** — either upload a CV as a PDF (text is extracted automatically with `pypdf`), or
   use the built-in sample resume text.
2. **Schema** — `Education`, `Experience`, and `CandidateProfile` Pydantic models define the
   exact shape of the expected output.
3. **Prompt** — a `ChatPromptTemplate` combines the resume text with format instructions
   generated automatically from the schema.
4. **LLM call** — the prompt is sent to Google Gemini (`gemini-3.6-flash`) via
   `langchain-google-genai`.
5. **Parsing** — the model's response is parsed and validated into a `CandidateProfile` object,
   then converted to indented JSON.

## Setup

### 1. Get a free Gemini API key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey) and sign in with a Google
   account (free, no credit card required).
2. Click **Create API Key** and copy it.

### 2. Provide the key to the notebook

- **On Google Colab (recommended):** open the Secrets panel (🔑 icon in the left sidebar), add a
  secret named `GOOGLE_API_KEY` with your key, and enable "Notebook access". The notebook reads
  it automatically.
- **Locally / other environments:** set it as an environment variable before launching Jupyter:
  ```bash
  export GOOGLE_API_KEY="your-key-here"
  ```
  If neither is set, the notebook will prompt you to paste the key manually at runtime.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

(On Colab this is handled by the `%pip install` cell at the top of the notebook.)

## Usage

Open `hr_candidate_profile_parser.ipynb` and run the cells in order. When prompted, either
upload your own CV as a PDF or let it fall back to the built-in sample resume.

The final parsed profile is printed as JSON and also saved to `candidate_profile.json`.

## Example output

```json
{
  "full_name": "John Smith",
  "email": "john.smith@email.com",
  "education": [
    { "degree": "B.Sc. Computer Science", "institution": "MIT", "year": 2020 }
  ],
  "skills": ["Python", "Machine Learning", "Data Analysis"],
  "experience": [
    { "role": "Software Engineer", "company": "Google", "years": "2020-2023" },
    { "role": "Data Scientist", "company": "OpenAI", "years": "2023-Present" }
  ]
}
```

## Notes

- The notebook uses `temperature=0` for consistent, deterministic extraction.
- `email` and `education[].year` are optional fields — the pipeline won't fail if a resume is
  missing them.
- Do not commit any real CV files or API keys to this repository.
