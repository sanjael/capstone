import sys, torch, torch.nn.functional as F, numpy as np, math
sys.path.insert(0, '.')

from services.model_loader import get_classification_model, get_device
from services.preprocessing import preprocess_for_inference
from PIL import Image

DEVICE = get_device()
model  = get_classification_model()

img  = Image.open(r'../Testing/pituitary/1201.jpg').convert('RGB')
tensor, _ = preprocess_for_inference(img, size=224)
x = tensor.to(DEVICE)

# Exactly replicate the predictor.py classification block
tumor_type = None
probs = None
max_prob = 0.0
entropy = None

try:
    logits = model(x)
    raw_logits_np = logits.detach().cpu().numpy()[0]
    print("step1: logits ok", raw_logits_np)

    logits_np = raw_logits_np.astype(np.float64)
    logits_np = (logits_np - logits_np.mean()) / (logits_np.std() + 1e-8)
    logits_norm = torch.tensor(logits_np, dtype=torch.float32).unsqueeze(0)
    print("step2: norm ok")

    T = 1.5
    probs = F.softmax(logits_norm / T, dim=1).detach().cpu().numpy()[0]
    print("step3: softmax ok", probs)

    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum()
    print("step4: clip ok", probs)

    classes = ["glioma", "meningioma", "pituitary"]
    max_prob = float(np.max(probs))
    predicted_class = classes[int(np.argmax(probs))]
    print("step5: class ok", predicted_class, max_prob)

    entropy_raw = -np.sum(probs * np.log(probs))
    entropy = float(entropy_raw / math.log(len(classes)))
    print("step6: entropy ok", entropy)

    tumor_type = predicted_class
    print("step7: tumor_type ok", tumor_type)

except Exception as e:
    import traceback
    print("EXCEPTION AT STEP:", e)
    traceback.print_exc()

print("\n--- FINAL ---")
print("tumor_type:", tumor_type)
print("entropy:", entropy)
