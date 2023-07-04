import torch 

# # BYOL Config
model_config = dict(
    batch_size=512,
    show_batch=False,
    show_batch_size=10,
    EPOCHS = 200,
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu',
    SAVE_PLOTS = True,
    VISUALIZE_PLOTS = False,
    SAVE_DIR = "./moco/run",
    MODEL_NAME = "SimSiam",
    WEIGHTS = None,
    OPTIMIZER = "SGD",
    VALIDATION_FREQ = 3,
    HIDDEN_SIZE = 4096,
    EMBEDDING_SIZE = 128,
    PROJECTION_SIZE = 128,
    RESUME = False,
    RESUME_DIR= "./moco/run18",
    MOMENTUM=0.9,
    LEARNING_RATE=0.06,
    WEIGHT_DECAY = 5e-4,
    WARM_UP = 0,
    dataset = "CIFAR10",
    Description = "MOCO Implementation without Projection Layer"
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