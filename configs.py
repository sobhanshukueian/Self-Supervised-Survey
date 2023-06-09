import torch 

# # BYOL Config
model_config = dict(
    batch_size=50,
    show_batch=False,
    show_batch_size=10,
    EPOCHS = 800,
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu',
    VERBOSE = 2,
    SAVE_PLOTS = True,
    VISUALIZE_PLOTS = False,
    SAVE_DIR = "./mocooo/run",
    MODEL_NAME = "MOCOOO",
    WEIGHTS = None,
    OPTIMIZER = "AdamW",
    EVALUATION_FREQ = 1,
    HIDDEN_SIZE = 4096,
    EMBEDDING_SIZE = 256,
    RESUME = False,
    RESUME_DIR= "./moco/run18",
    MOMENTUM=0.99,
    LEARNING_RATE=0.002,
    WEIGHT_DECAY = 0.00005,
    USE_SCHEDULER = False,
    WARM_UP = 0,
    Description = "All terms with activation"
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