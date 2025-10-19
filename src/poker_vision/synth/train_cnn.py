import os, glob, cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

RANKS = ['A','K','Q','J','10','9','8','7','6','5','4','3','2']
SUITS = ['\u2660','\u2665','\u2666','\u2663']
W=64; H=64

def load_images(folder):
  paths = glob.glob(os.path.join(folder, "*.png"))
  X,y = [], []
  for p in paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    X.append(img)
    base = os.path.basename(p).split("_")[0]
    y.append(base)
  return np.array(X), np.array(y)

class DS(Dataset):
  def __init__(self, X,y,labels):
    self.X = X.astype(np.float32)/255.0
    self.y = np.array([labels.index(lbl) for lbl in y], dtype=np.int64)
  def __len__(self): return len(self.X)
  def __getitem__(self,i):
    return torch.from_numpy(self.X[i][None,:,:]), torch.tensor(self.y[i])

class Net(nn.Module):
  def __init__(self, n):
    super().__init__()
    self.c1=nn.Conv2d(1,16,3,padding=1)
    self.c2=nn.Conv2d(16,32,3,padding=1)
    self.c3=nn.Conv2d(32,64,3,padding=1)
    self.fc1=nn.Linear(64*(W//8)*(H//8),128)
    self.fc2=nn.Linear(128,n)
  def forward(self,x):
    x=F.max_pool2d(F.relu(self.c1(x)),2)
    x=F.max_pool2d(F.relu(self.c2(x)),2)
    x=F.max_pool2d(F.relu(self.c3(x)),2)
    x=x.view(x.size(0),-1)
    x=F.relu(self.fc1(x))
    return self.fc2(x)

def train_head(X,y,labels,epochs=6):
  Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=0.1,random_state=42,stratify=y)
  tr=DS(Xtr,ytr,labels); va=DS(Xva,yva,labels)
  dl_tr=DataLoader(tr,batch_size=128,shuffle=True)
  dl_va=DataLoader(va,batch_size=256)
  net=Net(len(labels))
  opt=torch.optim.Adam(net.parameters(),lr=1e-3)
  for ep in range(epochs):
    net.train()
    for xb,yb in dl_tr:
      opt.zero_grad(); loss=F.cross_entropy(net(xb), yb); loss.backward(); opt.step()
  return net

def main():
  Xr,yr = load_images("poker_vision/synth_data/rank")
  Xs,ys = load_images("poker_vision/synth_data/suit")
  rank_net = train_head(Xr, yr, RANKS, epochs=6)
  ys_int = np.array([str(v) for v in ys])
  suit_labels = [str(ord(c)) for c in SUITS]
  suit_net = train_head(Xs, ys_int, suit_labels, epochs=6)
  dummy = torch.randn(1,1,H,W)
  torch.onnx.export(rank_net, dummy, "rank.onnx", input_names=["input"], output_names=["rank_logits"], opset_version=18)
  torch.onnx.export(suit_net, dummy, "suit.onnx", input_names=["input"], output_names=["suit_logits"], opset_version=18)
  print("rank.onnx suit.onnx")


if __name__ == "__main__":
  main()
