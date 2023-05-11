import torch 

# First Config
model_config = dict(
    batch_size=512,
    show_batch=False,
    show_batch_size=10,
    EPOCHS = 800,
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu',
    VERBOSE = 2,
    SAVE_PLOTS = True,
    VISUALIZE_PLOTS = False,
    SAVE_DIR = "/content/drive/MyDrive/byol/run",
    MODEL_NAME = "BYOL Model Adaptive...",
    WEIGHTS = None,
    OPTIMIZER = "SGD",
    KNN_EVALUATION_PERIOD = 3,
    HIDDEN_SIZE = 4096,
    EMBEDDING_SIZE = 256,
    RESUME = False,
    RESUME_DIR= "./runs/byol/run4",
    MOMENTUM=0.9,
    LEARNING_RATE=0.4,
    WEIGHT_DECAY = 0.00005,
    USE_SCHEDULER = True
)