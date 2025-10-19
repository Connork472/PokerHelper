import json, cv2, numpy as np
from mss import mss

class ROISelector:
  def __init__(self, config_path="roi_config.json"):
    self.config_path=config_path; self.cfg={}; self.drag=False; self.start=None; self.name=None; self.box=None; self.sct=mss(); self.mon=None
  def _save(self):
    with open(self.config_path,"w") as f: json.dump(self.cfg,f,indent=2)
  def _load(self):
    try:
      with open(self.config_path) as f: self.cfg=json.load(f)
    except:
      self.cfg={}
  def pick_monitor(self):
    self.mon=self.sct.monitors[1]; return self.mon
  def run(self):
    self._load(); mon=self.pick_monitor()
    win="Screen h=hand b=board a=amounts s=save q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    def mouse(evt,x,y,flags,ud):
      if evt==cv2.EVENT_LBUTTONDOWN:
        self.drag=True; self.start=(x,y)
      elif evt==cv2.EVENT_MOUSEMOVE and self.drag:
        self.box=(min(self.start[0],x),min(self.start[1],y),abs(x-self.start[0]),abs(y-self.start[1]))
      elif evt==cv2.EVENT_LBUTTONUP:
        self.drag=False
        if self.name and self.box and self.box[2]>5 and self.box[3]>5:
          self.cfg[self.name]={"left":self.box[0]+mon["left"],"top":self.box[1]+mon["top"],"width":self.box[2],"height":self.box[3]}
          self.box=None; self.name=None
    cv2.setMouseCallback(win, mouse)
    while True:
      frame=np.array(self.sct.grab(mon))[:,:,:3]
      disp=frame.copy()
      for k,r in self.cfg.items():
        cv2.rectangle(disp,(r["left"]-mon["left"], r["top"]-mon["top"]),(r["left"]-mon["left"]+r["width"], r["top"]-mon["top"]+r["height"]),(0,255,255),2)
        cv2.putText(disp,k,(r["left"]-mon["left"], r["top"]-mon["top"]-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
      if self.box is not None:
        x,y,w,h=self.box; cv2.rectangle(disp,(x,y),(x+w,y+h),(0,255,0),2)
      cv2.imshow(win, disp)
      key=cv2.waitKey(1)&0xFF
      if key==ord('q'): break
      elif key==ord('h'): self.name="my_hand_region"
      elif key==ord('b'): self.name="board_region"
      elif key==ord('a'): self.name="amounts_region"
      elif key==ord('s'): self._save()
    self._save(); cv2.destroyAllWindows()

if __name__=="__main__":
  ROISelector().run()