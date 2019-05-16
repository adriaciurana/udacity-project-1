import os
import torch
class Config:
	"""
		PATHS
	"""
	CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
	PROJECT_PATH = os.path.dirname(CURRENT_PATH)
	
	ENVS_PATH = os.path.join(PROJECT_PATH, "envs")
	CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "checkpoints")
	MODELS_PATH = os.path.join(PROJECT_PATH, "models")
	
	BANANA_ENV_PATH = os.path.join(ENVS_PATH, "Banana_Linux", "Banana.x86_64")
	CHECKPOINT_BANANA_PATH = os.path.join(CHECKPOINT_PATH, "banana")
	MODEL_BANANA_PATH = os.path.join(MODELS_PATH, "banana")

	BANANA_PIXELS_ENV_PATH = os.path.join(ENVS_PATH, "VisualBanana_Linux", "Banana.x86_64")
	CHECKPOINT_PIXELS_BANANA_PATH = os.path.join(CHECKPOINT_PATH, "banana_pixels")
	MODEL_PIXELS_BANANA_PATH = os.path.join(MODELS_PATH, "banana_pixels")

	PRETRAINED_MOBILENET_V2_1 = os.path.join(CURRENT_PATH, "mobilenetv2", "mobilenetv2_1.0-0c6065bc.pth")
	PRETRAINED_MOBILENET_V2_05 = os.path.join(CURRENT_PATH, "mobilenetv2", "mobilenetv2_0.5-eaa6f9ad.pth")
	PRETRAINED_MOBILENET_V2_01 = os.path.join(CURRENT_PATH, "mobilenetv2", "mobilenetv2_0.1-7d1d638a.pth")


	"""
		TORCH CONFIG
	"""
	DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	"""
		 TRAINING PARAMETERS
	"""
	UPDATE_EVERY = 4
	LR = .001
	BATCH_SIZE = 8
	GAMMA = 0.99
	TAU = 1e-3

	BUFFER_A = 0.7
	BUFFER_B = 0.5
	BUFFER_EPS = 0.01
	BUFFER_SIZE = int(1e5)
