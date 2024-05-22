import torch
import torch
import numpy as np
import torch
from tqdm.auto import tqdm
assert torch.cuda.is_available()
from torch2trt import torch2trt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modele_weigth = 'quantized_classifier_weigth.pt'
model = torch.load('FireResNet50-100PK.pt').to(device)

example_input = torch.randn(1, 3, 224, 224).cuda()


model_trt = torch2trt(model,[example_input], fp16_mode=True)

torch.save(model_trt.state_dict(), modele_weigth)

model.load_state_dict(modele_weigth)
torch.save(model, 'result')
