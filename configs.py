import torch 

# # BYOL Config
model_config = dict(
    batch_size=512,
    show_batch=False,
    show_batch_size=10,
    EPOCHS = 800,
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu',
    SAVE_PLOTS = True,
    VISUALIZE_PLOTS = False,
    SAVE_DIR = "./moco_var",
    MODEL_NAME = "MOCO_VAR",
    WEIGHTS = None,
    OPTIMIZER = "AdamW",
    VALIDATION_FREQ = 3,
    HIDDEN_SIZE = 4096,
    EMBEDDING_SIZE = 128,
    PROJECTION_SIZE = 128,
    RESUME = True,
    RESUME_DIR= "/kaggle/input/self-superised-survey/Self-Supervised-Survey/moco_var/weights",
    MOMENTUM=0.9,
    LEARNING_RATE=0.002,
    WEIGHT_DECAY = 5e-4,
    WARM_UP = 0,
    dataset = "CIFAR10",
    Backbone = "resnet18",
    Description = "MOCO_VAR Implementation "
)

# Linear Evaluation Config
# model_config = dict(
#     batch_size=512,
#     show_batch=True,
#     show_batch_size=10,
#     EPOCHS = 100,
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu',
#     VERBOSE = 2,
#     SAVE_PLOTS = True,
#     VISUALIZE_PLOTS = False,
#     SAVE_DIR = "/content/drive/MyDrive/linear_eval/run",
#     MODEL_NAME = "Linear Classifier Evaluation",
#     # WEIGHTS = "D:/Ai/Projects/self-supervised-learning/sss/weights/best_BYOL.pt",
#     WEIGHTS=None,
#     OPTIMIZER = "SGD",
#     KNN_EVALUATION_PERIOD = 3,
#     HIDDEN_SIZE = 4096,
#     EMBEDDING_SIZE = 256,
#     RESUME = False,
#     RESUME_DIR= "D:/Ai/Projects/self-supervised-learning/sss/weights/best_BYOL.pt",
#     MOMENTUM=0.9,
#     LEARNING_RATE=0.4,
#     WEIGHT_DECAY = 0,
#     USE_SCHEDULER = False
# )
