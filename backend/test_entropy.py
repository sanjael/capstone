import sys, traceback, torch, torch.nn.functional as F, numpy as np, math
sys.path.insert(0, '.')

from services.model_loader import get_classification_model, get_device
from services.preprocessing import preprocess_for_inference
from PIL import Image

DEVICE = get_device()
model  = get_classification_model()

img  = Image.open(r'../Testing/pituitary/1201.jpg').convert('RGB')
tensor, _ = preprocess_for_inference(img, size=224)
x = tensor.to(DEVICE)

try:
    logits = model(x)
    print('logits shape :', logits.shape)
    print('logits values:', logits.detach().cpu().numpy())

    raw = logits.detach().cpu().numpy()[0].astype(np.float64)
    norm = (raw - raw.mean()) / (raw.std() + 1e-8)
    logits_norm = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)

    T = 1.5
    probs = F.softmax(logits_norm / T, dim=1).detach().cpu().numpy()[0]
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum()

    print('probs        :', probs)

    entropy_raw = -np.sum(probs * np.log(probs))
    entropy     = float(entropy_raw / math.log(3))
    print('entropy      :', entropy)
    print('prediction_entropy:', round(entropy, 4))

except Exception:
    traceback.print_exc()
