"""
Screen Analyzer - Captures and analyzes poker table from screen

Provides:
- Card detection using existing ONNX classifiers
- Button/UI detection with OCR fallback
- Pot size and bet amount parsing
- Region management for different poker clients
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import time
import json
import os
import sys
import re

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'poker_vision'))

try:
    from mss import mss
except ImportError:
    mss = None

# Try to import card detection components
try:
    from classify.infer_onnx_two import TwoHead
except ImportError:
    try:
        from classify.infer_onnx_fallback import TwoHead
    except ImportError:
        TwoHead = None

try:
    from geometry.card_finder import find_cards
except ImportError:
    find_cards = None


@dataclass
class Region:
    """A screen region for detection"""
    x: int
    y: int
    width: int
    height: int
    name: str = ""
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
    
    def to_mss_monitor(self) -> dict:
        return {
            "top": self.y,
            "left": self.x,
            "width": self.width,
            "height": self.height
        }


@dataclass
class ScreenAnalyzerConfig:
    """Configuration for screen analyzer"""
    # Card regions
    hand_region: Optional[Region] = None
    board_region: Optional[Region] = None
    
    # UI regions
    pot_region: Optional[Region] = None
    fold_button: Optional[Region] = None
    check_button: Optional[Region] = None
    call_button: Optional[Region] = None
    raise_button: Optional[Region] = None
    bet_input: Optional[Region] = None
    
    # Detection settings
    card_confidence_threshold: float = 0.3
    ocr_confidence_threshold: float = 0.5
    
    # Update rate
    capture_interval: float = 0.5  # seconds between captures


class ScreenAnalyzer:
    """
    Analyzes poker table from screen captures.
    
    Combines:
    - Card detection (existing ONNX classifiers)
    - OCR for text (pot size, bet amounts, button labels)
    - Region-based detection
    """
    
    def __init__(self, config: Optional[ScreenAnalyzerConfig] = None):
        self.config = config or ScreenAnalyzerConfig()
        
        # Initialize screen capture
        self.sct = mss() if mss else None
        
        # Initialize card classifier
        self.card_classifier = None
        self._init_card_classifier()
        
        # Initialize OCR
        self.ocr_engine = None
        self._init_ocr()
        
        # State
        self.last_capture_time = 0
        self.last_hand_cards: List[str] = []
        self.last_board_cards: List[str] = []
        self.last_pot_size: float = 0
        
    def _init_card_classifier(self) -> None:
        """Initialize the card classifier"""
        if TwoHead is None:
            print("Warning: Card classifier not available")
            return
            
        try:
            config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
            rank_path = os.path.join(config_dir, 'rank.onnx')
            suit_path = os.path.join(config_dir, 'suit.onnx')
            
            if os.path.exists(rank_path) and os.path.exists(suit_path):
                self.card_classifier = TwoHead(rank_path, suit_path)
            else:
                print(f"Warning: ONNX model files not found in {config_dir}")
        except Exception as e:
            print(f"Warning: Could not initialize card classifier: {e}")
    
    def _init_ocr(self) -> None:
        """Initialize OCR engine"""
        try:
            import pytesseract
            self.ocr_engine = pytesseract
        except ImportError:
            print("Warning: pytesseract not available. Install with: pip install pytesseract")
            print("Also need to install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture the full screen"""
        if not self.sct:
            return None
            
        try:
            monitor = self.sct.monitors[1]  # Primary monitor
            screenshot = self.sct.grab(monitor)
            return np.array(screenshot)[:, :, :3]  # RGB only
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None
    
    def capture_region(self, region: Region) -> Optional[np.ndarray]:
        """Capture a specific region of the screen"""
        if not self.sct or not region:
            return None
            
        try:
            screenshot = self.sct.grab(region.to_mss_monitor())
            return np.array(screenshot)[:, :, :3]
        except Exception as e:
            print(f"Region capture error: {e}")
            return None
    
    def detect_cards_in_region(
        self, 
        region: Region, 
        max_cards: int = 5
    ) -> List[str]:
        """
        Detect cards in a screen region.
        
        Args:
            region: Screen region to analyze
            max_cards: Maximum number of cards to detect
            
        Returns:
            List of card strings (e.g., ['As', 'Kh'])
        """
        if not self.card_classifier:
            return []
            
        roi = self.capture_region(region)
        if roi is None:
            return []
        
        detected_cards = []
        
        # Method 1: Contour-based detection
        if find_cards:
            try:
                cards = find_cards(roi, min_area=800, aspect_min=0.5, aspect_max=0.9)
                for warped_card, quad in cards[:max_cards]:
                    if warped_card.size == 0:
                        continue
                    label, conf = self.card_classifier.predict_with_conf(
                        warped_card, 
                        self.config.card_confidence_threshold
                    )
                    if conf > 0:
                        detected_cards.append(label)
            except Exception:
                pass
        
        # Method 2: Grid-based detection (fallback)
        if len(detected_cards) < max_cards:
            grid_cards = self._detect_cards_grid(roi, max_cards)
            for card in grid_cards:
                if card not in detected_cards:
                    detected_cards.append(card)
        
        return detected_cards[:max_cards]
    
    def _detect_cards_grid(self, roi: np.ndarray, max_cards: int) -> List[str]:
        """Grid-based card detection"""
        if not self.card_classifier:
            return []
            
        h, w = roi.shape[:2]
        cards = []
        
        # Divide ROI into columns
        cols = min(max_cards, 5)
        card_width = w // cols
        
        for i in range(cols):
            x_start = i * card_width
            x_end = min((i + 1) * card_width, w)
            
            if x_end - x_start < 30:
                continue
            
            card_roi = roi[:, x_start:x_end]
            
            try:
                label, conf = self.card_classifier.predict_with_conf(
                    card_roi,
                    self.config.card_confidence_threshold
                )
                if conf > 0:
                    cards.append(label)
            except Exception:
                pass
        
        return cards
    
    def detect_hand_cards(self) -> List[str]:
        """Detect hole cards"""
        if not self.config.hand_region:
            return []
        
        cards = self.detect_cards_in_region(self.config.hand_region, max_cards=2)
        self.last_hand_cards = cards
        return cards
    
    def detect_board_cards(self) -> List[str]:
        """Detect community cards"""
        if not self.config.board_region:
            return []
        
        cards = self.detect_cards_in_region(self.config.board_region, max_cards=5)
        self.last_board_cards = cards
        return cards
    
    def detect_pot_size(self) -> float:
        """Detect pot size using OCR"""
        if not self.config.pot_region or not self.ocr_engine:
            return 0.0
        
        roi = self.capture_region(self.config.pot_region)
        if roi is None:
            return 0.0
        
        try:
            # Preprocess for OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Run OCR
            text = self.ocr_engine.image_to_string(thresh, config='--psm 7')
            
            # Parse number from text
            pot = self._parse_amount(text)
            self.last_pot_size = pot
            return pot
            
        except Exception as e:
            print(f"OCR error: {e}")
            return 0.0
    
    def _parse_amount(self, text: str) -> float:
        """Parse a monetary amount from text"""
        # Remove currency symbols and whitespace
        text = text.strip().replace('$', '').replace(',', '').replace(' ', '')
        
        # Find numbers (including decimals)
        match = re.search(r'[\d.]+', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return 0.0
    
    def detect_button_text(self, region: Region) -> str:
        """Detect text on a button using OCR"""
        if not self.ocr_engine:
            return ""
        
        roi = self.capture_region(region)
        if roi is None:
            return ""
        
        try:
            # Preprocess
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            text = self.ocr_engine.image_to_string(thresh, config='--psm 7')
            return text.strip().lower()
            
        except Exception:
            return ""
    
    def find_buttons_by_text(self, screen: np.ndarray) -> Dict[str, Region]:
        """
        Find action buttons by searching for text.
        
        Returns dict mapping button type to region.
        """
        if not self.ocr_engine:
            return {}
        
        buttons = {}
        button_keywords = {
            'fold': ['fold'],
            'check': ['check'],
            'call': ['call'],
            'raise': ['raise', 'bet'],
            'all_in': ['all-in', 'allin', 'all in']
        }
        
        # This is a simplified approach - in practice would need
        # more sophisticated text location
        try:
            # Get all text with bounding boxes
            data = self.ocr_engine.image_to_data(
                screen, 
                output_type=self.ocr_engine.Output.DICT
            )
            
            for i, text in enumerate(data['text']):
                text_lower = text.lower().strip()
                if not text_lower:
                    continue
                
                for button_type, keywords in button_keywords.items():
                    if any(kw in text_lower for kw in keywords):
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        
                        # Expand region slightly for clicking
                        buttons[button_type] = Region(
                            x - 10, y - 10,
                            w + 20, h + 20,
                            button_type
                        )
                        break
            
        except Exception as e:
            print(f"Button detection error: {e}")
        
        return buttons
    
    def analyze_table(self) -> Dict:
        """
        Perform full table analysis.
        
        Returns dict with all detected information.
        """
        # Rate limit captures
        current_time = time.time()
        if current_time - self.last_capture_time < self.config.capture_interval:
            return {
                'hand_cards': self.last_hand_cards,
                'board_cards': self.last_board_cards,
                'pot_size': self.last_pot_size
            }
        
        self.last_capture_time = current_time
        
        # Detect everything
        result = {
            'hand_cards': self.detect_hand_cards(),
            'board_cards': self.detect_board_cards(),
            'pot_size': self.detect_pot_size(),
            'timestamp': current_time
        }
        
        return result
    
    def set_region(self, name: str, region: Tuple[int, int, int, int]) -> None:
        """Set a region by name"""
        x, y, w, h = region
        reg = Region(x, y, w, h, name)
        
        if name == 'hand':
            self.config.hand_region = reg
        elif name == 'board':
            self.config.board_region = reg
        elif name == 'pot':
            self.config.pot_region = reg
        elif name == 'fold':
            self.config.fold_button = reg
        elif name == 'check':
            self.config.check_button = reg
        elif name == 'call':
            self.config.call_button = reg
        elif name == 'raise':
            self.config.raise_button = reg
        elif name == 'bet_input':
            self.config.bet_input = reg
    
    def save_config(self, filepath: str) -> None:
        """Save region configuration to file"""
        config_dict = {
            'card_confidence_threshold': self.config.card_confidence_threshold,
            'capture_interval': self.config.capture_interval,
            'regions': {}
        }
        
        region_names = ['hand_region', 'board_region', 'pot_region',
                       'fold_button', 'check_button', 'call_button',
                       'raise_button', 'bet_input']
        
        for name in region_names:
            region = getattr(self.config, name, None)
            if region:
                config_dict['regions'][name] = {
                    'x': region.x,
                    'y': region.y,
                    'width': region.width,
                    'height': region.height
                }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self, filepath: str) -> None:
        """Load region configuration from file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.config.card_confidence_threshold = config_dict.get(
            'card_confidence_threshold', 0.3
        )
        self.config.capture_interval = config_dict.get('capture_interval', 0.5)
        
        for name, region_data in config_dict.get('regions', {}).items():
            region = Region(
                region_data['x'],
                region_data['y'],
                region_data['width'],
                region_data['height'],
                name
            )
            setattr(self.config, name, region)
    
    def select_region_interactive(self, title: str = "Select Region") -> Optional[Tuple[int, int, int, int]]:
        """
        Interactive region selection using OpenCV.
        
        Returns (x, y, width, height) or None if cancelled.
        """
        screen = self.capture_screen()
        if screen is None:
            return None
        
        # Create selection window
        window_name = f"{title} - Click and drag, ENTER to confirm, ESC to cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        # Selection state
        drawing = False
        start_point = None
        end_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, end_point
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                end_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_point = (x, y)
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            display = screen.copy()
            
            # Draw instruction overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
            cv2.putText(display, f"Select {title}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if start_point and end_point:
                cv2.rectangle(display, start_point, end_point, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                if start_point and end_point:
                    x1, y1 = start_point
                    x2, y2 = end_point
                    x, y = min(x1, x2), min(y1, y2)
                    w, h = abs(x2 - x1), abs(y2 - y1)
                    
                    if w > 10 and h > 10:
                        cv2.destroyAllWindows()
                        return (x, y, w, h)
            elif key == 27:  # Escape
                cv2.destroyAllWindows()
                return None
        
        return None

