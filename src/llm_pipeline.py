import os
import time
from tqdm import tqdm
from utils import Log_Experiment
from config import OPENAI_API_KEY, HUGGINGFACE_HUB_TOKEN
from typing import List, Tuple

# Optional imports — wrap them so code still runs if not installed
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
    from transformers import StoppingCriteria, StoppingCriteriaList
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset
    import torch
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    GenerationConfig = None
    torch = None

# For HF hosted inference (lightweight, hosted)
try:
    from huggingface_hub import login
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

# RAG / embeddings
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    SentenceTransformer = None
    chromadb = None
    embedding_functions = None

class TQDMProgress(StoppingCriteria):
    def __init__(self, total_tokens):
        self.progress_bar = tqdm(total=total_tokens, desc="Generating tokens")
        self.generated_tokens = 0

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids.shape[-1] = tokens generated so far
        new_tokens = input_ids.shape[-1] - self.generated_tokens
        if new_tokens > 0:
            self.progress_bar.update(new_tokens)
            self.generated_tokens += new_tokens
        return False  # continue generation

class LLM_Pipeline:
    # ---------------------------
    # PROMPTS (prompt engineering)
    # ---------------------------
    BASE_PROMPT = """You are a fraud detection assistant. Answer in exactly three lines:
    LABEL: FRAUD or NOT_FRAUD
    CONFIDENCE: decimal between 0 and 1
    REASON: short explanation sentence.

    Transaction:
    {txn_text}

    Now provide the answer adhering to the format exactly.
    """

    FEW_SHOT_PROMPT = """You are a fraud detection assistant. Answer in exactly three lines:
    LABEL: FRAUD or NOT_FRAUD
    CONFIDENCE: decimal between 0 and 1
    REASON: short explanation sentence.

    Example 1:
    Transaction: Transaction TID: 100 — Amount: $5 — Device: usual
    LABEL: NOT_FRAUD
    CONFIDENCE: 0.05
    REASON: Small routine purchase.

    Example 2:
    Transaction: Transaction TID: 101 — Amount: $2000 — Device: new
    LABEL: FRAUD
    CONFIDENCE: 0.95
    REASON: Very large unusual purchase on a new device.

    Now classify:
    Transaction:
    {txn_text}

    Answer now.
    """

    @staticmethod
    def huggingface_login():
        try:
            login(token=HUGGINGFACE_HUB_TOKEN)
        except Exception as e:
            raise("Warning: failed to login to Hugging Face Hub:", e)

    @staticmethod
    def load_local_model(model_name:str, device:str = None, use_4bit=False):
        """
        Load TinyLlama or Mistral 1B model for CPU inference.
        """
        print(f"Loading lightweight model {model_name} on {device}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=HUGGINGFACE_HUB_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": "cpu"} ,
            token=HUGGINGFACE_HUB_TOKEN
        )

        model.eval()
        return tokenizer, model, device
    
    @staticmethod
    def local_generate(tokeniser, model, prompt:str, max_new_token:int=32, temperature:float = 0.0, device: str = None) -> str:
        """
        CPU-friendly generation.
        """
        inputs = tokeniser(prompt, return_tensors="pt")

        progress_callback = TQDMProgress(max_new_token)
        stopping_criteria = StoppingCriteriaList([progress_callback])

        with torch.inference_mode():  # disables gradients, speeds up CPU
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_token,
                stopping_criteria=stopping_criteria,
                #do_sample=False if temperature == 0.0 else True,
                do_sample=False,
                temperature=temperature,
                top_p=0.95 if temperature > 0 else 1.0,
                pad_token_id=tokeniser.eos_token_id
            )
        text = tokeniser.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()
    
    @staticmethod
    def call_hf_inference_api(prompt:str, hf_model:str, hf_token:str=None, max_tokens: int=150, temperature: float = 0.0):
        """
        Use the Hugging Face Inference API (InferenceClient) if available.
        Requires HUGGINGFACE_HUB_TOKEN env var or hf_token passed.
        """

        if InferenceClient is None:
            raise ImportError("Install huggingface_hub to use HF Inference API: pip install huggingface-hub")

        token = hf_token or HUGGINGFACE_HUB_TOKEN
        client=InferenceClient(token=token)
        response=client.text_generation(model=hf_model, inputs=prompt, max_new_tokens=max_tokens, temperature=temperature)

        if isinstance(response, dict) and "generated_text" in response:
            return response["generated_text"]
        if isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]: 
            return response[0]["generated_text"]
        return str(response)
    
    @staticmethod
    def build_rag_index(docs: List[Tuple[str, str]], persist_dir:str="vector_store",  embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Build a Chroma vector store from docs: list of (id, text).
        Stores each doc's id inside 'metadatas' so queries can return it.

        Returns the chroma collection object.
        """
        if SentenceTransformer is None or chromadb is None:
            raise ImportError("Install sentence-transformers and chromadb to use RAG: pip install sentence-transformers chromadb")

        # create embeddings model
        embedder = SentenceTransformer(embedding_model_name)

        ids = [str(d[0]) for d in docs]
        texts = [d[1] for d in docs]

        # compute embeddings as numpy array
        embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # instantiate chroma client & (re)create collection with persist dir as name
        client = chromadb.Client()
        # If a collection with that name exists, drop and recreate to avoid duplicates
        collection_name = persist_dir
        try:
            # Try to delete existing collection with same name to avoid duplicate records
            if collection_name in [c['name'] for c in client.list_collections()]:
                client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = client.create_collection(name=collection_name)

        # Build metadatas list to store ids so we can retrieve them later
        metadatas = [{'doc_id': i} for i in ids]

        # Add documents, embeddings and metadatas
        # Note: chroma's add() signature may vary; this is a general approach
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            metadatas=metadatas
        )

        # Try to persist if possible (some chroma installs have persist)
        try:
            collection.persist()
        except Exception:
            # not all chroma setups require explicit persist()
            pass

        return collection

    @staticmethod
    def query_rag(collection, query_text:str, top_k:int=3, embedding_model_name: str="all-MiniLM-L6-v2"):
        """
        Query the Chroma collection for top_k similar docs.
        Returns list of tuples: (doc_id, doc_text, distance) where doc_id may be None if not found.
        Works with Chroma versions that accept 'query_embeddings' or 'queries' (fallback).
        """

        if SentenceTransformer is None:
            raise ImportError("Install sentence-transformers")

        # encode query
        embedder = SentenceTransformer(embedding_model_name)
        q_emb = embedder.encode([query_text], convert_to_numpy=True)[0]

        # Query with modern param name first, fall back to older API name
        res = None
        try:
            # Newer Chroma versions use query_embeddings
            res = collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        except TypeError:
            # Older versions may accept 'queries' instead of 'query_embeddings'
            res = collection.query(
                queries=[q_emb.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

        docs = []
        if not res:
            return docs

        # Expected structure: res['documents'][0] is list of doc texts,
        # res['metadatas'][0] is list of metadata dicts corresponding to docs,
        # res['distances'][0] is list of distances.
        documents_list = res.get('documents', [[]])[0]
        metadatas_list = res.get('metadatas', [[]])[0]
        distances_list = res.get('distances', [[]])[0]

        # Build results; try to extract id from metadata's 'doc_id' (we saved that above)
        for i, doc_text in enumerate(documents_list):
            doc_id = None
            if i < len(metadatas_list) and isinstance(metadatas_list[i], dict):
                # our build_rag_index stored {'doc_id': id}
                doc_id = metadatas_list[i].get('doc_id') or metadatas_list[i].get('id') or metadatas_list[i].get('source')
            dist = distances_list[i] if i < len(distances_list) else None
            docs.append((doc_id, doc_text, dist))
        return docs
    
    # @staticmethod
    # def run_local_llm_on_df(df, engine="local_mistral", model_identifier=None, prompt_type="base", 
    #                         use_rag=False, rag_collection=None, tokenizer_model=None,
    #                         max_new_tokens=32, temperature=0.0, device=None, use_4bit=False, batch_size=4):
    #     """
    #     Run a local HF model (Mistral / Llama) or HF Inference API on a DataFrame containing
    #     ['transaction_id','txn_text','label'] and log outputs to experiment log.
    #     engine options:
    #     - 'local_mistral' -> use a local HF Mistral model id passed in model_identifier
    #     - 'local_llama'  -> use a local Llama model id passed in model_identifier
    #     - 'hf_inference' -> use HF Inference API (model_identifier is HF model id)
    #     If use_rag=True, rag_collection must be provided (from build_rag_index).
    #     """
    
    #     rows_to_log=[]
    #     local_tokeniser=local_model=None
    
    #     if engine in ("local_mistral", "local_llama"):
    #         if model_identifier is None:
    #             raise ValueError("model_identifier (HF model id or path) required for local models")
    #         local_tokeniser, local_model, used_devicd = LLM_Pipeline.load_local_model(model_identifier,device=device, use_4bit=use_4bit)

    #     #for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM run ({engine})"):
    #     for idx, row in tqdm(range(0, len(df), batch_size), desc="LLM run ({engine})"):
    #         txn_text = row["txn_text"]
    #         prompt_ctx=""
    #         if use_rag and rag_collection is not None:
    #             try:
    #                 retrieved = LLM_Pipeline.query_rag(rag_collection, txn_text, top_k=3)
    #                 if len(retrieved)>0:
    #                     prompt_ctx += "Retrieved similar transactions (for context):\n"
    #                     for i, (rid, doc, _) in enumerate(retrieved):
    #                         prompt_ctx += f"Example {i+1}: {doc}\n"
    #                     prompt_ctx += "\n"
    #             except Exception as e:
    #                 prompt_ctx += ""
    #                 raise e
                    
    #         prompt_template=LLM_Pipeline.FEW_SHOT_PROMPT if prompt_type == "few_shot" else LLM_Pipeline.BASE_PROMPT
    #         prompt = prompt_ctx + prompt_template.format(txn_text=txn_text)

    #         raw_resp=None
    #         try:
    #             if engine == "hf_inference":
    #                 raw_resp = LLM_Pipeline.call_hf_inference_api(prompt, hf_model=model_identifier)
    #             elif engine in ("local_mistral", "local_llama"):
    #                 out = LLM_Pipeline.local_generate(local_tokeniser, local_model, prompt, max_new_token=max_new_tokens, temperature=temperature, device=device)
    #                 # Local_generate returns generated suffix; combine or keep raw
    #                 raw_resp = out
    #             else:
    #                 raise ValueError("Unknown engine")
    #         except Exception as e:
    #             raw_resp = f"ERROR: {e}"

    #         # Parse LLM response (our standard parser)
    #         label_str, conf, reason = Log_Experiment.safe_parse_llm_response(raw_resp)
    #         normalized_label = 'FRAUD' if label_str and str(label_str).strip().lower().startswith('fraud') else 'NOT_FRAUD'
    #         conf_val = float(conf) if conf is not None else 0.5

    #         log_row = {
    #             'transaction_id': row['transaction_id'],
    #             'true_label': int(row['label']),
    #             'predicted_label': normalized_label,
    #             'predicted_prob': conf_val,
    #             'model_name': f"LOCAL_{engine.upper()}_{model_identifier.replace('/','_')}",
    #             'prompt_variant': prompt_type,
    #             'reason': reason or raw_resp,
    #             'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    #         }
    #         rows_to_log.append(log_row)

    #     # append rows to experiment CSV
    #     Log_Experiment.append_experiment_log(rows_to_log)
    #     return rows_to_log

    @staticmethod    
    def run_local_llm_on_df(
        df, engine="local_mistral", model_identifier=None, prompt_type="base", 
        use_rag=False, rag_collection=None, tokenizer_model=None,
        max_new_tokens=32, temperature=0.0, device=None, use_4bit=False, batch_size=4
    ):
        """
        Run a local HF model (Mistral / LLaMA) on a DataFrame containing
        ['transaction_id','txn_text','label'] with batching and tqdm progress.
        
        Args:
            df: pandas DataFrame
            engine: 'local_mistral', 'local_llama', 'hf_inference'
            model_identifier: HF model ID or local path
            prompt_type: 'base' or 'few_shot'
            use_rag: whether to use RAG retrieval
            rag_collection: Chroma collection if use_rag=True
            max_new_tokens, temperature: generation params
            device: CPU only in your case
            batch_size: number of transactions per batch
        
        Returns:
            List of logged dicts for experiment CSV
        """

        rows_to_log = []
        local_tokeniser = local_model = None

        # Load local model if required
        if engine in ("local_mistral", "local_llama"):
            if model_identifier is None:
                raise ValueError("model_identifier required for local models")
            local_tokeniser, local_model, used_device = LLM_Pipeline.load_local_model(
                model_identifier, device=device, use_4bit=False  # CPU only
            )

        # Process in batches
        num_batches = (len(df) + batch_size - 1) // batch_size

        for batch_start in tqdm(range(0, len(df), batch_size), desc=f"LLM run ({engine})"):
            batch_df = df.iloc[batch_start: batch_start + batch_size]
            prompts = []

            # Build prompts for batch
            for _, row in batch_df.iterrows():
                txn_text = row["txn_text"]
                prompt_ctx = ""
                if use_rag and rag_collection is not None:
                    try:
                        retrieved = LLM_Pipeline.query_rag(rag_collection, txn_text, top_k=3)
                        if len(retrieved) > 0:
                            prompt_ctx += "Retrieved similar transactions (for context):\n"
                            for i, (rid, doc, _) in enumerate(retrieved):
                                prompt_ctx += f"Example {i+1}: {doc}\n"
                            prompt_ctx += "\n"
                    except Exception as e:
                        prompt_ctx += ""
                        raise e

                prompt_template = LLM_Pipeline.FEW_SHOT_PROMPT if prompt_type == "few_shot" else LLM_Pipeline.BASE_PROMPT
                prompt = prompt_ctx + prompt_template.format(txn_text=txn_text)
                prompts.append(prompt)

            # Generate outputs for the batch
            raw_responses = []
            counter=1
            for prompt in prompts:
                print(f"\nProcess Counter: {counter}")
                try:
                    if engine == "hf_inference":
                        resp = LLM_Pipeline.call_hf_inference_api(prompt, hf_model=model_identifier)
                    elif engine in ("local_mistral", "local_llama"):
                        resp = LLM_Pipeline.local_generate(
                            local_tokeniser,
                            local_model,
                            prompt,
                            max_new_token=max_new_tokens,
                            temperature=temperature,
                            device=device
                        )
                    else:
                        raise ValueError("Unknown engine")
                except Exception as e:
                    resp = f"ERROR: {e}"

                raw_responses.append(resp)
                counter = counter + 1

            # Parse and log batch results
            for (_, row), raw_resp in zip(batch_df.iterrows(), raw_responses):
                label_str, conf, reason = Log_Experiment.safe_parse_llm_response(raw_resp)
                normalized_label = 'FRAUD' if label_str and str(label_str).strip().lower().startswith('fraud') else 'NOT_FRAUD'
                conf_val = float(conf) if conf is not None else 0.5

                log_row = {
                    'transaction_id': row['transaction_id'],
                    'true_label': int(row['label']),
                    'predicted_label': normalized_label,
                    'predicted_prob': conf_val,
                    'model_name': f"LOCAL_{engine.upper()}_{model_identifier.replace('/', '_')}",
                    'prompt_variant': prompt_type,
                    'reason': reason or raw_resp,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                rows_to_log.append(log_row)

        # Append rows to experiment CSV
        Log_Experiment.append_experiment_log(rows_to_log)
        return rows_to_log
    
    # Transfer Knowledge
    @staticmethod
    def fine_tune_llm_on_fraud_data(model_name, train_df, output_dir="models/fine_tuned_llm", num_train_epochs=1):
        """
        Fine-tune a small Hugging Face model (TinyLlama / Mistral-1B / Flan-T5)
        on your labeled fraud dataset using text prompts.

        Args:
            model_name (str): HF model ID, e.g. 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
            train_df (pd.DataFrame): must contain columns ['txn_text','label']
            output_dir (str): folder to save fine-tuned weights
            num_train_epochs (int): how many epochs to train (1–3 on CPU)
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"\n🔹 Starting fine-tuning for {model_name} on CPU")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

        # Prepare dataset: convert each row to simple prompt-response pairs
        def format_prompt(example):
            prompt = f"Transaction:\n{example['txn_text']}\n\nIs this fraud? "
            label_text = "FRAUD" if example["label"] == 1 else "NOT_FRAUD"
            return {"text": f"{prompt}{label_text}"}

        dataset = Dataset.from_pandas(train_df)
        dataset = dataset.map(format_prompt)

        tokenized_ds = dataset.map(
            lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=256),
            batched=True,
            remove_columns=dataset.column_names
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=2,     # CPU-friendly
            learning_rate=5e-5,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            fp16=False,
            dataloader_num_workers=0
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=data_collator
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"✅ Fine-tuned model saved to {output_dir}")
        return output_dir