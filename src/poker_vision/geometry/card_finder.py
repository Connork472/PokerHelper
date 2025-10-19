import cv2, numpy as np

def _order(pts):
  r = np.zeros((4,2),dtype="float32")
  s = pts.sum(axis=1); r[0]=pts[np.argmin(s)]; r[2]=pts[np.argmax(s)]
  d = np.diff(pts,axis=1); r[1]=pts[np.argmin(d)]; r[3]=pts[np.argmax(d)]
  return r

def warp(bgr, quad, out_size=(300,420)):
  rect = _order(quad.astype("float32"))
  w,h = out_size
  dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype="float32")
  M = cv2.getPerspectiveTransform(rect,dst)
  return cv2.warpPerspective(bgr,M,(w,h))

def find_cards(bgr_roi, min_area=1500, aspect_min=0.60, aspect_max=0.80):
  out=[]
  gray=cv2.cvtColor(bgr_roi,cv2.COLOR_BGR2GRAY)
  blur=cv2.GaussianBlur(gray,(5,5),0)
  th=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,10)
  cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  for c in cnts:
    area=cv2.contourArea(c)
    if area<min_area: continue
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*peri,True)
    if len(approx)!=4: continue
    x,y,w,h=cv2.boundingRect(approx)
    aspect=w/float(h) if h else 0
    if not(aspect_min<=aspect<=aspect_max): continue
    quad=approx.reshape(-1,2).astype("float32")
    warped=warp(bgr_roi,quad,(300,420))
    out.append((warped,quad))
  return out
