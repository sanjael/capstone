import urllib.request
import json
import os

# Find first test tumor image
test_dir = "dataset/detection/test/tumor"
img_file = next(f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
img_path = os.path.join(test_dir, img_file)
print(f"Test image: {img_path}")

boundary = "BOUNDARY_NEUROSCAN"
with open(img_path, "rb") as f:
    img_data = f.read()

body  = f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode()
body += img_data + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    "http://localhost:8001/predict/",
    data=body,
    method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)

try:
    r    = urllib.request.urlopen(req)
    data = json.loads(r.read())
except Exception as e:
    print(f"Request failed: {e}")
    raise

# Show key fields (skip huge base64 blobs)
skip = {"heatmap_image", "heatmap_gradcam", "heatmap_eigencam"}
summary = {k: v for k, v in data.items() if k not in skip}
print(json.dumps(summary, indent=2))

# Validate all required fields
print("\n=== FIELD VALIDATION ===")
checks = {
    "tumor_detected is bool":       isinstance(data.get("tumor_detected"), bool),
    "confidence in [0,1]":          0.0 <= float(data.get("confidence", -1)) <= 1.0,
    "uncertainty non-negative":     float(data.get("uncertainty", -1)) >= 0,
    "reliability valid":            data.get("reliability") in ["HIGH", "MEDIUM", "LOW"],
    "risk_level valid":             data.get("risk_level") in ["None", "Low", "Moderate", "High", "Critical"],
    "clinical_note non-empty":      bool(data.get("clinical_note", "")),
    "recommendation non-empty":     bool(data.get("recommendation", "")),
    "heatmap starts data:image":    (data.get("heatmap_image") or "").startswith("data:image"),
}

all_pass = True
for k, v in checks.items():
    status = "PASS" if v else "FAIL"
    if not v:
        all_pass = False
    print(f"  [{status}] {k}")

print()
print("PREDICTION ENDPOINT:", "PASS" if all_pass else "FAIL")
