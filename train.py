import evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from peft import get_peft_model, TaskType, LoraConfig
from peft import prepare_model_for_kbit_training
from accelerate import DistributedDataParallelKwargs

import json
import click
import wandb
from pathlib import Path
from tqdm import tqdm
import numpy as onp
from PIL import Image
from einops import rearrange, reduce, repeat, pack, unpack
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from transformers import BitsAndBytesConfig

CACHE = Path("<YOUR CACHE DIR>")
DATASET = Path("<PATH TO DATASET FOLDER>")

TRAIN = DATASET / "train_parsed_results/all_parsed.json"
EVAL = DATASET / "parsed_results/all_parsed.json"
GQA_TRAIN = DATASET / "GQA/train.json"
GQA_EVAL = DATASET / "GQA/test.json"

IMDIR = Path("<PATH TO IMAGES FOLDER>")

class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.labels = []
        self.categories = []

    def update(self, predictions, labels, categories):
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        self.categories.extend(categories)

    def clean(self, texts:list[str]):
        return [ 
            t.replace("iteself", "itself").replace("</s>","").split("objects:")[-1].strip().lower()
            for t in texts 
        ]

    def compute(self):
        metrics = {}
        
        self.predictions = self.clean(self.predictions)
        self.labels = self.clean(self.labels)

        categories, subcategories = [], []
        for category, label in zip(self.categories, self.labels):
            [subcategory, *_] = category.split(".")
            categories.append(category), subcategories.append(subcategory)

        metrics["accuracy"] = onp.mean([ p == l for p, l in zip(self.predictions, self.labels) ])
        metrics["problematic"] = onp.mean([ p == l for p, l in zip(self.predictions, self.labels) if l == "the question itself is problematic" ])
        metrics["non-problematic"] = onp.mean([ p == l for p, l in zip(self.predictions, self.labels) if l != "the question itself is problematic" ])

        for category in set(categories):
            metrics[category] = onp.mean([ p == l for p, l, c in zip(self.predictions, self.labels, categories) if c == category ])
        for subcategory in set(subcategories):
            metrics[subcategory] = onp.mean([ p == l for p, l, s in zip(self.predictions, self.labels, subcategories) if s == subcategory ])

        return metrics
        

class Template:
    prompt = None
    answer = None

    def parse(response:str) -> str:
        raise NotImplementedError

class BLIPTemplate(Template):
    prompt = "Question: {question} Short Answer:"
    answer = "{answer}</s>"

    def parse(response:str) -> str:
        return response.split("Short Answer:")[-1].strip()
    
class BLIPNoCodeTemplate(Template):
    prompt = """
    Given an image, the user asks a question and assistant answers it.
    USER:
    {question}
    ASSISTANT:
    """

    answer = "{answer}</s>"

    def parse(response:str) -> str:
        return response.split("ASSISTANT:")[-1].strip().lower()

class BLIPCodeTemplate(Template):
    prompt = """
    Given an image, the user asks a question and assistant executes the code and logs the results step-by-step to provide an answer.
    USER:
    {question}
    Code
    {codes}
    ASSISTANT:
    Log"""

    answer = """
    {logs}
    Answer:
    {answer}</s>
    """

    def parse(response:str) -> str:
        return response.split("Answer:")[-1].strip().lower()
    
class LLaVATemplate(Template):
    prompt = """
    USER: <image>
    Answer the question using a single word or phrase.
    {question}
    ASSISTANT:
    """

    answer = "{answer}</s>"

    def parse(response:str) -> str:
        return response.split("ASSISTANT:")[-1].strip().lower()


class LLaVACodeTemplate(Template):
    prompt = """
    USER: <image>
    Executes the code and logs the results step-by-step to provide an answer to the question.
    Question
    {question}
    Code
    {codes}
    ASSISTANT:
    Log
    """

    answer = """
    {logs}
    Answer:
    {answer}</s>
    """

    def parse(response:str) -> str:
        return response.split("Answer:")[-1].strip().lower()
    

class AGQA(Dataset):
    def __init__(self, file:Path, imdir:Path, template:Template):
        self.imdir = imdir
        self.template = template

        with open(file) as f: 
            self.data = json.load(f)

        self.qid2data = { sample["questionId"]:sample for sample in self.data }
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        questionId = torch.tensor(sample["questionId"])
        question, answer = sample["question"], sample["answer"], 
        
        image = Image.open(self.imdir / f"{sample['imageId']}.jpg")
        image = image.convert("RGB")
        
        codes, logs = [], []
        if ('codes' in sample) and ('logs' in sample):
            codes, logs = sample['codes'], sample['logs']
        elif 'codegen' in sample:
            for line in sample["codegen"].split("\n"):
                [code, log] = line.split(" ### ")
                codes.append(code.strip()), logs.append(log.strip())
            
        prompts = self.template.prompt.format(question=question, codes=codes)
        answers = self.template.answer.format(answer=answer, logs=logs)
        
        return {
            "questionId":questionId,
            "prompts":prompts, 
            "answers":answers,
            "images":image,
        }
    
    def get(self, qid:int):
        return self.qid2data[qid] 

def dataloaders(dataset, processor:AutoProcessor, template:Template, batch_size=32, training=True, workers=16, mask=True):
    cfg = {
        "padding":"longest",
        "return_tensors":"pt",
        "max_length":496, # 512 - 16
        "truncation":True
    }
        
    def collate_fn(batch):
        qids = torch.stack([b["questionId"] for b in batch])

        tokens = processor.tokenizer(
            [[b["prompts"], b["answers"]] for b in batch], 
            return_token_type_ids=True, **cfg)
        prompts = processor(
            images=[b["images"] for b in batch], 
            text=[b["prompts"] for b in batch], **cfg)
        inputs = processor(
            images=[b["images"] for b in batch],
            text=[[b["prompts"], b["answers"]] for b in batch], 
            **cfg)

        labels = inputs.input_ids.clone()
        if mask:
            labels[tokens.token_type_ids == 0] = -100
        
        return qids, inputs, prompts, labels
    
    if isinstance(dataset, torch.utils.data.Dataset):
        return DataLoader(
            dataset,
            batch_size=batch_size, 
            shuffle=training,
            drop_last=training,
            collate_fn=collate_fn, 
            num_workers=workers
        )

def train(model, loader:DataLoader, optimizer:Optimizer, scheduler:CosineAnnealingLR, epochs:int, checkpoint:Path, accelerator:Accelerator, skip:int=0):
    model.train() 
    
    for _ in range(epochs):
        for idx, (qid, inputs, prompts, labels) in tqdm(enumerate(loader), total=len(loader)):
            if idx < skip: continue
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**inputs, labels=labels)
                accelerator.backward(outputs.loss)
                accelerator.log({ "train.nll": outputs.loss.item() })
                optimizer.step()
                scheduler.step()
                
            if idx % 8192 == 42 and checkpoint:
                accelerator.save_state(str(CACHE/checkpoint))
            
    return model

def test(model, testloader, processor, template:Template, outdir:Path, accelerator:Accelerator):
    model.eval()
    _model = accelerator.unwrap_model(model)

    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    
    for idx, (qid, inputs, prompts, labels) in tqdm(enumerate(testloader), total=len(testloader)):
        qid = qid.cpu().numpy().tolist()
            
        with torch.inference_mode():
            outputs = _model.generate(
                **prompts,
                max_new_tokens=256,
            )

        prompts = processor.tokenizer.batch_decode(prompts.input_ids, skip_special_tokens=True)
        predictions = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        labels[labels == -100] = processor.tokenizer.pad_token_id
        labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        for i, qid in enumerate(qid):
            with open(outdir/f"{qid}.json", "w") as f:
                predictions[i] = predictions[i].replace(prompts[i], "")
                sample = testloader.dataset.get(qid)
                json.dump({ 
                    "prompts":prompts[i], 
                    "predictions":template.parse(predictions[i]),
                    "labels":template.parse(labels[i]),
                    "category":sample["category"],
                    "raw":predictions[i]
                }, f)

    metric = Metric()

    for file in tqdm(list(outdir.glob("*.json"))):
        with open(file) as f:
            data = json.load(f)
            metric.update([data["predictions"]], [data["labels"]], [data["category"]])
            
    accelerator.log(metric.compute())
        
def prepare_model_and_template(name:str, task:str, lora:bool=False):
    cfg = { "torch_dtype":torch.bfloat16, "low_cpu_mem_usage":True, "cache_dir":CACHE }
    if "llava" in name:
        model = LlavaForConditionalGeneration.from_pretrained(name, **cfg)
        processor = AutoProcessor.from_pretrained(name)
        mapping = { "standard":LLaVATemplate, "code":LLaVACodeTemplate }
        template = mapping[task]
        target_modules = ["q_proj", "v_proj"]
    elif "blip2" in name:
        model = Blip2ForConditionalGeneration.from_pretrained(name, **cfg)
        processor = Blip2Processor.from_pretrained(name)
        mapping = { "standard":BLIPTemplate, "nocode":BLIPNoCodeTemplate, "code":BLIPCodeTemplate }
        template = mapping[task]
        target_modules = ["q", "v"]
    elif "instructblip" in name:
        model = InstructBlipForConditionalGeneration.from_pretrained(name, **cfg)
        processor = InstructBlipProcessor.from_pretrained(name)
        mapping = { "standard":BLIPTemplate, "nocode":BLIPNoCodeTemplate, "code":BLIPCodeTemplate }
        template = mapping[task]
        target_modules = ["q_proj", "v_proj"]
    else:
        raise ValueError(f"Invalid model {name}")
    
    if lora:
        config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=target_modules,
            task_type="CAUSAL_LM" if "llava" in name else None,
        )
        model = get_peft_model(model, config)

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    return model, processor, template

def choose_dataset(dataset_name:str, mode:str, template:Template):
    assert mode in ['train', 'test']
    if dataset_name == 'GQA':
        if mode == 'train': return AGQA(GQA_TRAIN, IMDIR, template)
        if mode == 'test': return AGQA(GQA_EVAL, IMDIR, template)
    elif dataset_name == 'VISREAS':
        if mode == 'train': return AGQA(TRAIN, IMDIR, template)
        if mode == 'test': return AGQA(EVAL, IMDIR, template)
    

@click.command()
@click.option("--model", default="llava-hf/llava-1.5-13b-hf", type=str)
@click.option("--task", default="standard", type=str)
@click.option("--dataset_name", default="GQA", type=str)
@click.option("--training", default=True, type=bool)
@click.option("--testing", default=True, type=bool)
@click.option("--lora", default=True, type=bool)
@click.option("--lr", default=2e-5, type=float)
@click.option("--epochs", default=1, type=int)
@click.option("--batch-size", default=8, type=int)
@click.option("--workers", default=16, type=int)
@click.option("--checkpoint", default=None, type=Path)
@click.option("--outdir", default=None, type=Path)
@click.option("--mask", default=True, type=bool)
@click.option("--skip", default=0, type=int)
def main(model, task, dataset_name, training, testing, lora, lr, epochs, batch_size, workers, checkpoint, outdir, mask, skip):    
    accelerator = Accelerator(
        mixed_precision="bf16", gradient_accumulation_steps=4, split_batches=True, 
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        log_with="wandb"
    )
    accelerator.init_trackers(dataset_name)

    model, processor, template = prepare_model_and_template(model, task, lora)
    model = accelerator.prepare(model)
    
    trainset = choose_dataset(dataset_name, "train", template)
    trainloader = dataloaders(trainset, processor, template, training=training, batch_size=batch_size, workers=workers, mask=mask)
    testset = choose_dataset(dataset_name, "test", template)
    testloader = dataloaders(testset, processor, template, training=False, batch_size=batch_size, workers=workers, mask=mask)

    if training:
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 128, epochs * len(trainloader))

        trainloader, optimizer, scheduler = accelerator.prepare(trainloader, optimizer, scheduler)
        if skip > 0 and checkpoint:
            accelerator.load_state(str(CACHE/checkpoint))
        model = train(
            model, trainloader, optimizer, scheduler, 
            epochs=epochs, checkpoint=checkpoint, accelerator=accelerator, skip=skip
        )

        if checkpoint:accelerator.save_state(str(CACHE/checkpoint))
        del optimizer, scheduler, trainloader

    # test
    if testing:
        testloader = accelerator.prepare(testloader)
        if not training and checkpoint:
            accelerator.load_state(str(CACHE/checkpoint))
        test(model, testloader, processor, template=template, outdir=outdir, accelerator=accelerator)

    accelerator.end_training()
    
if __name__ == "__main__":
    main()