"""
Configurações globais para o projeto de classificação de leucemia
"""
import os
import torch

# Diretorios
TRAIN_DIR = 'dataset/cnmc2019_splits/images/train'
VAL_DIR = 'dataset/cnmc2019_splits/images/val'
TEST_DIR = 'dataset/cnmc2019_splits/images/test'
SAVE_DIR = 'models/modular'
FEATURES_DIR = 'features'

# Criar diretórios se não existirem
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# Hiperparâmetros
SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# Modelos CNN para treinar
CNN_MODELS = [
    "tf_efficientnetv2_b3"
    #"resnet50",
    #"efficientnet_b3.ra2_in1k" 
    

    #"densenet169.ra_in1k" 
    #"convnextv2_base.fcmae_ft_in22k_in1k",
    #"swinv2_base_window12_192.ms_in22k" 
]



# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
CLASS_NAMES = ['hem', 'all']
NUM_CLASSES = 2