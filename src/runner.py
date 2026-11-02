import os
import json
import time
import pandas as pd
from tqdm import tqdm

from prompt_templates import (
    get_zero_shot_prompt,
    get_few_shot_prompt,
    get_cot_prompt
)

from dotenv import load_dotenv, find_dotenv
from groq import Groq
from google import genai
from google.genai import types

from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ZeroFewShotResponse(BaseModel):
    toxicity_score: float = Field(ge=0.0, le=1.0)

class CotResponse(BaseModel):
    reasoning: str
    toxicity_score: float = Field(ge=0.0, le=1.0)

class ToxicitySubtypes(BaseModel):
    severe_toxicity: float = Field(ge=0.0, le=1.0)
    obscene: float = Field(ge=0.0, le=1.0)
    threat: float = Field(ge=0.0, le=1.0)
    insult: float = Field(ge=0.0, le=1.0)
    identity_attack: float = Field(ge=0.0, le=1.0)
    sexual_explicit: float = Field(ge=0.0, le=1.0)

class ZeroFewShotSubtypesResponse(BaseModel):
    toxicity_score: float = Field(ge=0.0, le=1.0)
    subtypes: ToxicitySubtypes

class CotSubtypesResponse(BaseModel):
    reasoning: str
    toxicity_score: float = Field(ge=0.0, le=1.0)
    subtypes: ToxicitySubtypes

EXPERIMENTS = [
    {"model": "openai/gpt-oss-120b", "provider": "groq", "technique": "zero_shot"},
    {"model": "openai/gpt-oss-120b", "provider": "groq", "technique": "few_shot"},
    {"model": "openai/gpt-oss-120b", "provider": "groq", "technique": "cot"},
    {"model": "qwen/qwen3.8-27b", "provider": "groq", "technique": "zero_shot"},
    {"model": "qwen/qwen3.8-27b", "provider": "groq", "technique": "few_shot"},
    {"model": "qwen/qwen3.8-27b", "provider": "groq", "technique": "cot"},
    {"model": "models/gemini-3.1-flash-lite", "provider": "gemini", "technique": "zero_shot"},
    {"model": "models/gemini-3.1-flash-lite", "provider": "gemini", "technique": "few_shot"},
    {"model": "models/gemini-3.1-flash-lite", "provider": "gemini", "technique": "cot"}
]

def get_schema(technique, include_subtypes):
    if technique == "cot":
        return CotSubtypesResponse if include_subtypes else CotResponse
    else:
        return ZeroFewShotSubtypesResponse if include_subtypes else ZeroFewShotResponse

def call_llm(prompts, model_name, provider, technique, include_subtypes):
    schema_clss = get_schema(technique, include_subtypes)
    match provider:
        case "groq":
            response = groq_client.chat.completions.create(
                model = model_name,
                messages = prompts,
                temperature = 0.0,
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "toxicity_schema",
                        "schema": schema_clss.model_json_schema()
                    }
                }
            )
            return json.loads(response.choices[0].message.content)
        case "gemini":
            contents = []
            system_prompt = None
            for p in prompts:
                if p["role"] == "system":
                    system_prompt = p["content"]
                else:
                    contents.append(types.Content(
                        role="user" if p["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=p["content"])]
                    ))

            response = gemini_client.models.generate_content(
                model = model_name,
                contents = contents,
                config = types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=schema_clss,
                    system_instruction=system_prompt
                )
            )
            return json.loads(response.text)

def get_delay(provider, model_name, technique):
    if technique in ("few_shot", "cot"):
        return 11.5 if ("qwen" in model_name.lower() and technique == "cot") else 9.5

    match provider:
        case "groq":
            return 2.5
        case "gemini":
            return 4.5

def try_call_llm(prompts, model_name, provider, technique,  include_subtypes, max_retries=6):
    for attempt in range(1, max_retries+1):
        try:
            json_llm_output = call_llm(prompts, model_name, provider, technique, include_subtypes)
            json_llm_output["error"] = False
            print(f"[SUCCESS]: {model_name} from {provider}")
            time.sleep(get_delay(provider, model_name, technique))
            return json_llm_output
        except Exception as e:
            err_str = str(e).lower()

            if "daily" in err_str or "quota_exceeded" in err_str:
                print(f"[MODEL LIMITED]: daily limit reached for {provider}, stopping retries")
                break

            if "429" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str:
                wait_time = 60
                print(f"[RATE LIMITED]: {model_name} on {provider}, cooling down for {wait_time}s...")
            else:
                wait_time = (2 ** (attempt+1))
                print(f"[WARNING]: attempt {attempt}/{max_retries} failed: {e}")

            if(attempt == max_retries):
                break

            print(f"[RETRYING...]: waiting {wait_time}s")
            time.sleep(wait_time)

    return {
        "toxicity_score": -1,
        "reasoning": f"[ERROR]: failed after {max_retries} attempts, or model daily limit reached.",
        "subtypes": None if include_subtypes else {},
        "error": True
    }

def get_prompt_technique(technique, comment, include_subtypes):
    match technique:
        case "zero_shot":
            return get_zero_shot_prompt(comment, include_subtypes)
        case "few_shot":
            return get_few_shot_prompt(comment, include_subtypes)
        case "cot":
            return get_cot_prompt(comment, include_subtypes)

def run(sampled_df, include_subtypes):
    sub_tag = "subtypes" if include_subtypes else ""
    sampled_dict = sampled_df.to_dict('records')

    for exp in EXPERIMENTS:
        cln_model_name = exp["model"].replace("/", "_")
        output_file = f"results/llm_outputs/{cln_model_name}_{exp['provider']}_{exp['technique']}_{sub_tag}.json"
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    existing_data = json.load(f)
                if len(existing_data) >= 200:
                    print(f"[SKIPPING]: {output_file} is already complete (200/200 items).")
                    continue
            except Exception:
                pass

        results = []
        processed_ids = set()
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                results = json.load(f)
                processed_ids = {r["id"] for r in results if "id" in r}

        print(f"\n[RUNNING...]: {exp['model']} from {exp['provider']} | {exp['technique']} {sub_tag} ")
        for idx, row in enumerate(tqdm(sampled_dict)):
            if idx >= 1000:
                print("[STOPPED]: daily free tier limit reached.")
                break

            comment_id = row.get("id", idx)
            if comment_id in processed_ids:
                continue

            comment = row["comment_text"]
            prompts = get_prompt_technique(exp["technique"], comment, include_subtypes)
            llm_output = try_call_llm(prompts, exp["model"], exp["provider"], exp["technique"], include_subtypes)

            record = {
                "id": comment_id,
                "provider": exp["provider"],
                "model": exp["model"],
                "technique": exp["technique"],
                "comment_text": comment,
                "result": llm_output
            }
            results.append(record)

            if len(results) % 10 == 0:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

def main():
    filepath = "results/data/sampled_n_200.csv"
    sampled_df = pd.read_csv(filepath)
    run(sampled_df, False)
    run(sampled_df, True)

if __name__ == "__main__":
    main()