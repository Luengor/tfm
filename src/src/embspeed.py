from common.embeddings import EmbeddingModelNames, get_model
from PIL import Image
from tqdm import tqdm
from time import time
import torch

# Accelerate
if torch.accelerator.is_available():
    accelerator = torch.accelerator.current_accelerator()
    assert accelerator is not None
    torch.device(accelerator)

K = 1000 

def test_model(model_name: EmbeddingModelNames, image) -> float:
    # Get the model
    model = get_model(model_name)

    start = time()

    # Get the emebeddings from the image a lot of times
    for _ in tqdm(range(K), desc=f"Testing {model_name}"):
        model.get_embedding(image)

    end = time()

    return end - start

if __name__ == "__main__":
    image = Image.open("test.png").convert("RGB")

    data = {}
    for name in EmbeddingModelNames:
        data[name] = test_model(name, image)

    for name, duration in data.items():
        print(f"{name}: {duration:.2f} seconds for {K} runs")

