import pandas as pd
import openai
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from openai import OpenAI
from tqdm import tqdm
from utils.synthesis import generate_reasoning_prompt
from dotenv import load_dotenv
import sys
import time
import json
import re

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=os.environ.get(api_key),  
)

output_path = "data/valid_with_reasoning.csv"
output_path_jsonl = "data/valid_reasoning.jsonl"
unique_key = "input"

# Format dialogue into readable multiline format
def format_dialogue(text):
    tokens = re.split(r"(\[seeker\]|\[supporter\])", text)
    formatted = []
    for i in range(1, len(tokens), 2):
        speaker = tokens[i].strip("[]")
        if i + 1 < len(tokens):
            utterance = tokens[i + 1].strip()
            formatted.append(f"[{speaker}]: {utterance}")
    return "\n".join(formatted)

# Format reasoning into multiline by step
def format_reasoning(text):
    steps = []
    for line in text.split("Step "):
        if line.strip():
            steps.append("Step " + line.strip())
    return "\n".join(steps)

# Load the CSV file
df = pd.read_csv("data/valid.csv")
df["generated_reasoning"] = ""

# Check if the output file already exists and read it to avoid reprocessing
jsonl_done_keys = set()
if os.path.exists(output_path_jsonl):
    with open(output_path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                jsonl_done_keys.add(entry["dialogue"])
            except Exception:
                continue
    print(f"Resuming from checkpoint: {len(jsonl_done_keys)} samples already saved to JSONL.")
else:
    print("Starting fresh (no existing JSONL).")
    
if os.path.exists(output_path):
    df_done = pd.read_csv(output_path)
    done_keys = set(df_done[unique_key])
    print(f"Resuming from checkpoint. {len(done_keys)} rows already completed.")
else:
    df_done = pd.DataFrame()
    done_keys = set()
    print("Starting fresh.")

for i, row in tqdm(df.iterrows(), total=len(df)):
    if row[unique_key] in done_keys:
        continue  # skip already processed rows
    try:
        prompt = generate_reasoning_prompt(row)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            top_p=0.9,
            max_tokens=512
        )

        reasoning = response.choices[0].message.content.strip()
        row["generated_reasoning"] = reasoning

        pd.DataFrame([row]).to_csv(output_path, mode="a", index=False, header=not os.path.exists(output_path))
        
        formatted_dialogue = "\n".join([part.strip() for part in re.split(r"(\[seeker\]|\[supporter\])", row["input"]) if part.strip()])

        reasoning_steps = []
        for line in reasoning.split("Step "):
            if line.strip():
                reasoning_steps.append("Step " + line.strip())
        formatted_reasoning = "\n".join(reasoning_steps)
        
        json_record = {
            "problem_type": row["problem_type"],
            "emotion_type": row["emotion_type"],
            "dialogue": format_dialogue(row["input"]),
            "score": int(row["output"]),
            "reasoning": format_reasoning(reasoning)
        }
        
        with open(output_path_jsonl, "a", encoding="utf-8") as f:
            json_str = json.dumps(json_record, ensure_ascii=False, indent=2).replace("\\n", "\n")
            f.write(json_str + "\n")
            
        time.sleep(1.2)

    except Exception as e:
        print(f"Error at index {i}: {e}")
        time.sleep(5)
