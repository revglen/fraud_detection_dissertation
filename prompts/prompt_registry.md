# Prompt Registry - versions & templates

## base_prompt_v1
You are a fraud detection assistant. Answer in exactly three lines:
LABEL: FRAUD or NOT_FRAUD
CONFIDENCE: decimal between 0 and 1
REASON: short explanation sentence.

Transaction:
{txn_text}

Now provide the answer adhering to the format exactly.

## few_shot_v1
(Contains 2 short labeled examples then the transaction - see llm_pipeline.FEW_SHOT_PROMPT)