import torch
from torch import nn
from accelerate import Accelerator
from diffusers import FluxPipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5TokenizerFast
from transformers.modeling_outputs import BaseModelOutput

def check_decoder(prompt_embeds):

    model_name = "google/t5-v1_1-xxl"
    t_model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda:5")
    
    encoder_outputs = BaseModelOutput(last_hidden_state=prompt_embeds.to("cuda:5"))
    # Generate output text conditioned on prompt_embeds
    outputs = t_model.generate(
        encoder_outputs=encoder_outputs,
        max_length=12,
        num_beams=5,
    )

    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))


original_embeds = torch.load('/home/tiras/Nurit/A cat relaxing on a field of grass/new_embedding_30.pt')
check_decoder(original_embeds)

