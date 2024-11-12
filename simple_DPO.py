from datasets import load_dataset, Dataset, DatasetDict
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
import torch
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
import numpy as np
import gc
from peft import PeftModel

model_name = "Qwen/Qwen2-0.5B-Instruct" #meta-llama/Meta-Llama-3-8B
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
		model_name,
		low_cpu_mem_usage=True,
		torch_dtype=torch.bfloat16,
		load_in_4bit=True,  #True
		use_flash_attention_2=False, #attn_implementation="flash_attention_2",
		bnb_4bit_compute_dtype=torch.bfloat16,
		bnb_4bit_quant_type="nf4",
	)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
		lora_alpha=128,
		lora_dropout=0.05,
		r=64,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules=[
			"q_proj",
			"k_proj",
			"v_proj",
		],
	)

model = get_peft_model(model, peft_config)

train_dataset = load_dataset('trl-lib/ultrafeedback_binarized')['train']
val_dataset = load_dataset('trl-lib/ultrafeedback_binarized')['test']
# print(train_dataset[:2])

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation

def softmax(x1, x2):
	e_x = np.exp(x1)/(np.exp(x1) + np.exp(x2))
	# print(e_x)
	return e_x

def extract_prompt(example):
    r"""
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both
    the chosen and rejected completions.

    For more details, see [`maybe_extract_prompt`].
    """
    for idx in range(min(len(example["chosen"]), len(example["rejected"]))):
        if example["chosen"][idx] != example["rejected"][idx]:
            if example["chosen"][idx - 1] == " ":  # remove space before the prompt
                idx -= 1
            break

    # print(softmax(np.array(example['score_chosen']), np.array(example['score_rejected'])))
    return {
        "prompt": example["chosen"][0]['content'],
        "chosen": example["chosen"][1]['content'],
        "rejected": example["rejected"][1]['content'],
        "soft_label": softmax(np.array(example['score_chosen']), np.array(example['score_rejected']))
    }

# Helper function to format the dataset
def formatting_func(examples):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}

    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }
# formatted_dataset = DatasetDict({"train": train_dataset.map(formatting_func),
#                                  "test": val_dataset.map(formatting_func)})

train_dataset = train_dataset.map(extract_prompt)
val_dataset = val_dataset.map(extract_prompt)
print(train_dataset[:2])

formatted_dataset = DatasetDict({"train": train_dataset,
                                 "test": val_dataset})

# Configuring the training arguments
training_args = DPOConfig(
		num_train_epochs=1,
		per_device_train_batch_size=2, #16
		do_eval=True,
		per_device_eval_batch_size=1, #8
		adam_epsilon=1e-08,
		learning_rate=5e-06,
		lr_scheduler_type="linear",
		warmup_ratio=0.1,
		seed=42,
		logging_steps=100,
		save_steps=500,
		save_strategy="steps",
		output_dir="/clinical_nlp/uncertainty_quant/dpo_model_5e-6_1epoch",
		gradient_checkpointing=True,
		bf16=True,
		remove_unused_columns=False,
	)

dpo_trainer = DPOTrainer(
		model,
		args=training_args,
		beta=training_args.beta,
		train_dataset=formatted_dataset["train"],
		eval_dataset=formatted_dataset["test"],
		tokenizer=tokenizer,
		max_length=training_args.max_length,
		max_prompt_length=training_args.max_prompt_length,
		peft_config=peft_config,
	)

dpo_trainer.train()
dpo_trainer.model.save_pretrained("/clinical_nlp/uncertainty_quant/dpo_model_5e-6_1epoch_adapter")
tokenizer.save_pretrained("/clinical_nlp/uncertainty_quant/dpo_model_5e-6_1epoch_adapter")

del dpo_trainer, model
gc.collect()
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, return_dict=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = PeftModel.from_pretrained(base_model, "/clinical_nlp/uncertainty_quant/dpo_model_5e-6_1epoch_adapter")
model = model.merge_and_unload()

model.save_pretrained("/clinical_nlp/uncertainty_quant/dpo_model_5e-6_1epoch")
tokenizer.save_pretrained("/clinical_nlp/uncertainty_quant/dpo_model_5e-6_1epoch")
