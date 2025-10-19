from PIL import Image, ImageDraw, ImageFont
import numpy as np, cv2, os, random

RANKS = ['A','K','Q','J','10','9','8','7','6','5','4','3','2']
SUITS = ['\u2660','\u2665','\u2666','\u2663']
FONTS = [
  "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
  "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
  "/System/Library/Fonts/Supplemental/Courier New.ttf"
]
OUT = "poker_vision/synth_data"
W,H = 64,64

os.makedirs(f"{OUT}/rank", exist_ok=True)
os.makedirs(f"{OUT}/suit", exist_ok=True)

def pick_font(size):
  for p in FONTS:
    try:
      return ImageFont.truetype(p, size)
    except:
      pass
  return ImageFont.load_default()

def bg():
  c = random.randint(200,255)
  x = np.full((H,W,3), c, dtype=np.uint8)
  cv2.rectangle(x,(0,0),(W-1,H-1),(random.randint(210,255),)*3,1)
  return x

def render(t, color, scale):
  img = Image.fromarray(bg())
  draw = ImageDraw.Draw(img)
  fsize = max(14, int(H*scale))
  font = pick_font(fsize)
  bbox = draw.textbbox((0,0), t, font=font)
  tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
  x = max(0, (W-tw)//2)
  y = max(0, (H-th)//2)
  draw.text((x,y), t, font=font, fill=color)
  arr = np.array(img)
  if random.random()<0.6:
    arr = cv2.GaussianBlur(arr,(3,3),0)
  if random.random()<0.5:
    arr = cv2.convertScaleAbs(arr, alpha=random.uniform(0.85,1.15), beta=random.randint(-12,12))
  return arr

def main(n=800):
  for r in RANKS:
    for i in range(n):
      col = (0,0,0)
      im = render(r, col, random.uniform(0.6,0.9))
      cv2.imwrite(f"{OUT}/rank/{r}_{i}.png", im)
  for s in SUITS:
    for i in range(n):
      col = (0,0,0) if s in ['\u2660','\u2663'] else (220,0,0)
      im = render(s, col, random.uniform(0.6,0.95))
      cv2.imwrite(f"{OUT}/suit/{ord(s)}_{i}.png", im)

if __name__ == "__main__":
  main()
