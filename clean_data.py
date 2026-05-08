import re
import hashlib
import pandas as pd
from pathlib import Path

INPUT_FILE = "data/ai-medical-chatbot.csv"
FULL_OUTPUT = "data/cleaned_medical_full.csv"
SMALL_OUTPUT = "data/cleaned_medical_25mb.csv"
TARGET_MB = 25

# Patterns
_USELESS_PATTERNS = [
    r"For (further )?information consult a [\w\s]+ online\s*-->.*",
    r"For (further )?doubts consult a [\w\s]+ online\s*-->.*",
    r"For more information consult [\w\s]+ online\s*-->.*",
    r"Get back to [\w\s]+ online\s*-->.*",
    r"Revert with more information to [\w\s]+ online\s*-->.*",
    r"\(attachment removed to protect patient identity\)",
    r"attachment removed.*",
]

_USELESS_RE = re.compile("|".join(_USELESS_PATTERNS), flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)

_JUNK = re.compile(r"Â|â€™|â€œ|â€\\x9d|â€\"|\\u00e2|\\u0080|\\u0099|\\xa0|\\u0093|\\u0094")
_SPACE = re.compile(r"\s+")

# Cleaning Functions
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = _JUNK.sub(" ", text)
    text = _USELESS_RE.sub("", text)
    text = _SPACE.sub(" ", text).strip()

    return text

def row_hash(patient, doctor):
    combined = patient[:200] + doctor[:200]
    return hashlib.md5(combined.encode()).hexdigest()

def useful_answer(text):
    if len(text) < 30:
        return False

    bad = ["consult a", "online -->", "for more information", "i am here to help you",]
    lower = text.lower()
    if sum(p in lower for p in bad) >= 2 and len(text.split()) < 25:
        return False
    return True

def truncate(text, limit=2000):
    if len(text) <= limit:
        return text

    cut = text[:limit]
    last_period = cut.rfind(". ")

    if last_period > limit // 2:
        return cut[:last_period + 1]
    return cut

# Load Dataset
print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="latin1",
    on_bad_lines="skip"
)

print("Original rows:", len(df))

# Keep Needed Columns
keep_cols = ["Description", "Patient", "Doctor"]
df = df[keep_cols].copy()

# Remove Missing Rows
df.dropna(subset=["Patient", "Doctor"], inplace=True)

# Clean Text
for col in keep_cols:
    df[col] = df[col].astype(str).apply(clean_text)

# Remove Useless Answers 
df = df[df["Doctor"].apply(useful_answer)].copy()

# Remove Duplicates 
df["_hash"] = df.apply(lambda r: row_hash(r["Patient"], r["Doctor"]), axis=1)

df.drop_duplicates(subset="_hash", inplace=True)
df.drop(columns="_hash", inplace=True)

# Truncate Long Answers
df["Doctor"] = df["Doctor"].apply(truncate)

# Reset Index
df.reset_index(drop=True, inplace=True)

# SAVE FULL CLEANED DATASET
Path("data").mkdir(exist_ok=True)

df.to_csv(FULL_OUTPUT, index=False, encoding="utf-8")

full_size = Path(FULL_OUTPUT).stat().st_size / (1024 ** 2)

print("\nFULL CLEANED DATASET")
print("Rows:", len(df))
print(f"Size: {full_size:.2f} MB")
print("Saved:", FULL_OUTPUT)

# CREATE 25MB VERSION
sample_df = df.sample(n=min(20000, len(df)), random_state=42)

temp_path = "data/temp.csv"
sample_df.to_csv(temp_path, index=False)
sample_size = Path(temp_path).stat().st_size / (1024 ** 2)
estimated_rows = int((TARGET_MB / sample_size) * len(sample_df) * 0.97)

small_df = df.sample(n=min(estimated_rows, len(df)), random_state=42).reset_index(drop=True)
small_df.to_csv(SMALL_OUTPUT, index=False, encoding="utf-8")
small_size = Path(SMALL_OUTPUT).stat().st_size / (1024 ** 2)
Path(temp_path).unlink(missing_ok=True)

print("\n25MB DATASET")
print("Rows:", len(small_df))
print(f"Size: {small_size:.2f} MB")
print("Saved:", SMALL_OUTPUT)

print("\nDONE")
print("Full cleaned file :", FULL_OUTPUT)
print("25MB cleaned file :", SMALL_OUTPUT)