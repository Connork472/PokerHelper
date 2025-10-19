import cv2, numpy as np, onnxruntime as ort

def _softmax(x):
  e = np.exp(x - np.max(x, axis=1, keepdims=True))
  return e / np.sum(e, axis=1, keepdims=True)

class TwoHead:
  def __init__(self, rank_path="rank.onnx", suit_path="suit.onnx", providers=None):
    self.rank = ort.InferenceSession(rank_path, providers=providers or ["CPUExecutionProvider"])
    self.suit = ort.InferenceSession(suit_path, providers=providers or ["CPUExecutionProvider"])
    self.H = int(self.rank.get_inputs()[0].shape[2])
    self.W = int(self.rank.get_inputs()[0].shape[3])
    self.rank_labels = ['A','K','Q','J','10','9','8','7','6','5','4','3','2']
    self.suit_labels = ['s','h','d','c']

  def _prep(self, bgr):
    # Convert to grayscale
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Normalize lighting
    g = cv2.equalizeHist(g)
    
    # Apply Gaussian blur to reduce noise
    g = cv2.GaussianBlur(g, (3, 3), 0)
    
    # Resize with better interpolation
    g = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 1]
    g = g.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    return g[None, None, :, :]

  def _predict_logits(self, tile):
    x = self._prep(tile)
    r_logits = self.rank.run(None, {self.rank.get_inputs()[0].name: x})[0]
    s_logits = self.suit.run(None, {self.suit.get_inputs()[0].name: x})[0]
    return r_logits, s_logits

  def _predict_one_corner(self, card_bgr, corner="tl"):
    H, W = card_bgr.shape[:2]
    if corner == "tl":
      tile = card_bgr[0:int(0.32*H), 0:int(0.38*W)]
    else:
      tile = card_bgr[0:int(0.32*H), int(0.62*W):W]
    r_logits, s_logits = self._predict_logits(tile)
    r_prob = _softmax(r_logits)
    s_prob = _softmax(s_logits)
    r_i = int(np.argmax(r_prob, axis=1)[0])
    s_i = int(np.argmax(s_prob, axis=1)[0])
    
    # Improved confidence calculation - use geometric mean for better accuracy
    r_conf = float(np.max(r_prob))
    s_conf = float(np.max(s_prob))
    conf = np.sqrt(r_conf * s_conf)  # Geometric mean is more robust
    
    return self.rank_labels[r_i] + self.suit_labels[s_i], conf, tile

  def predict(self, card_bgr):
    a, ca, _ = self._predict_one_corner(card_bgr, "tl")
    b, cb, _ = self._predict_one_corner(card_bgr, "tr")
    return a if ca >= cb else b

  def predict_with_conf(self, card_bgr, confidence_threshold=0.55):
    """
    Predict card from BGR image with confidence threshold.
    Returns (label, confidence) tuple.
    If confidence is below threshold, returns ("", 0.0).
    """
    a, ca, _ = self._predict_one_corner(card_bgr, "tl")
    b, cb, _ = self._predict_one_corner(card_bgr, "tr")
    
    # Choose the higher confidence result
    if ca >= cb:
      label, conf = a, ca
    else:
      label, conf = b, cb
    
    # Apply confidence threshold
    if conf < confidence_threshold:
      return "", 0.0
    
    return label, conf

