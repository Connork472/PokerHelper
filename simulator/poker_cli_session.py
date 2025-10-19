#!/usr/bin/env python3
"""
poker_cli_session.py

Minimal local Monte Carlo for Texas Hold'em study:
- Outputs Win%, Tie%, Equity%, and Kelly% of bankroll (even-money) — no dollar inputs.
- You set your hand once; then add board cards across rounds (flop -> turn -> river).
- Defaults for players and trials; adjust anytime via commands.

Commands (type at the prompt):
  /help                 Show help
  /status               Show current settings and cards
  /players N            Set players (2..10)
  /trials N             Set default trials (>=1000)
  /hand <2 cards>       Set your hand (e.g., /hand As Kh)
  /add <cards>          Add board cards (e.g., /add 7h 2d 2s) — appends up to 5 total
  /clear                Clear board only (keep your hand)
  /new                  New hand: clears board and prompts for a new hand
  /quit                 Exit

Card formats accepted (mix & match, commas or spaces):
  As Kh   |  A s, K h  |  10c, Tc  |  ten clubs, ace spades  |  10 of clubs

Notes:
  • Kelly assumes even-money (b=1): f = max(0, 2p − 1), reported as % of bankroll.
  • This tool is for local/offline study. Do not automate against live sites.
"""

import re, random, math, sys
from treys import Card, Deck, Evaluator  # pyright: ignore[reportMissingImports]

# ---------- Config (defaults) ----------
DEFAULT_PLAYERS = 6
DEFAULT_TRIALS  = 200_000
RNG_SEED        = None  # set to an int for reproducibility across runs

# ---------- Parsing utilities ----------
RANK_MAP = {
    "2":"2","two":"2","3":"3","three":"3","4":"4","four":"4","5":"5","five":"5",
    "6":"6","six":"6","7":"7","seven":"7","8":"8","eight":"8","9":"9","nine":"9",
    "10":"T","t":"T","ten":"T","j":"J","jack":"J","q":"Q","queen":"Q",
    "k":"K","king":"K","a":"A","ace":"A",
}
SUIT_MAP = {"c":"c","club":"c","clubs":"c","d":"d","diamond":"d","diamonds":"d",
            "h":"h","heart":"h","hearts":"h","s":"s","spade":"s","spades":"s"}

def norm(s): return re.sub(r"\s+"," ", s.strip().lower())

def tokenize_cards(inp: str):
    """
    Accepts many friendly forms, returns list of normalized codes like 'As','Th'.
    """
    s = norm(inp.replace(",", " "))
    if not s:
        return []
    tokens = s.split()
    out = []; i = 0
    while i < len(tokens):
        tok = tokens[i]

        # compact like "As" / "10c" or with stray space "A s"
        m = re.match(r"^(10|[2-9]|[tjqka])\s*([cdhs])$", tok)
        if m:
            r_raw, s_raw = m.group(1), m.group(2)
            rank = RANK_MAP.get(r_raw, r_raw.upper().replace("10","T")).upper()
            suit = SUIT_MAP.get(s_raw, s_raw.lower())
            out.append(rank + suit); i += 1; continue

        # split form "ace spades" or "A s"
        if i+1 < len(tokens):
            r_raw, s_raw = tokens[i], tokens[i+1]
            if r_raw == "of" and i+2 < len(tokens):
                r_raw, s_raw = tokens[i+1], tokens[i+2]; i += 1
            rank = RANK_MAP.get(r_raw, r_raw.upper().replace("10","T"))
            suit = SUIT_MAP.get(s_raw, s_raw.lower())
            if rank in "23456789TJQKA" and suit in "cdhs":
                out.append(rank + suit); i += 2; continue

        # phrase "ten of clubs"
        if i+2 < len(tokens) and tokens[i+1] == "of":
            r_raw, s_raw = tokens[i], tokens[i+2]
            rank = RANK_MAP.get(r_raw, r_raw.upper().replace("10","T"))
            suit = SUIT_MAP.get(s_raw, s_raw.lower())
            if rank in "23456789TJQKA" and suit in "cdhs":
                out.append(rank + suit); i += 3; continue

        # fallback compact "10c" without spaces
        m2 = re.match(r"^(10|[2-9]|[tjqka])([cdhs])$", tok)
        if m2:
            r_raw, s_raw = m2.group(1), m2.group(2)
            rank = RANK_MAP.get(r_raw, r_raw.upper().replace("10","T")).upper()
            suit = SUIT_MAP.get(s_raw, s_raw.lower())
            out.append(rank + suit); i += 1; continue

        raise ValueError(f"Could not parse near: '{tok}'")
    return out

def to_treys(cards_str_list):
    return [Card.new(cs) for cs in cards_str_list]

# ---------- Math & simulation ----------
def kelly_even_money(equity: float) -> float:
    """Kelly for b=1 payoff (even-money): f = max(0, 2p - 1)."""
    return max(0.0, min(1.0, 2*equity - 1))

def simulate(players, my_cards, board_cards, trials, seed=None):
    if seed is not None:
        random.seed(seed)
    ev = Evaluator(); wins = ties = 0
    for _ in range(trials):
        deck = Deck()
        # remove known cards
        for c in my_cards + board_cards:
            deck.cards.remove(c)
        # deal opponents
        opps = [[deck.draw(1)[0], deck.draw(1)[0]] for _ in range(players - 1)]
        # complete board
        need = 5 - len(board_cards)
        sim_board = board_cards + deck.draw(need)
        # rank
        myr = ev.evaluate(sim_board, my_cards)
        best = min(ev.evaluate(sim_board, h) for h in opps)
        if myr < best: wins += 1
        elif myr == best: ties += 1
    win_p = wins / trials
    tie_p = ties / trials
    equity = win_p + 0.5 * tie_p
    return win_p, tie_p, equity

# ---------- Session loop ----------
HELP_TEXT = __doc__.split("Commands")[0] + "Commands" + __doc__.split("Commands")[1].split("Card formats")[0] + "Card formats" + __doc__.split("Card formats")[1].split("Notes")[0] + "Notes:\n  • Kelly assumes even-money (b=1): f = max(0, 2p − 1), reported as %.\n  • This tool is for local/offline study. Do not automate against live sites.\n"

def show_status(players, trials, my_cards_str, board_str):
    print("\n--- Status ---")
    print(f"Players: {players}")
    print(f"Trials:  {trials:,}")
    print(f"Hand:    {(' '.join(my_cards_str) if my_cards_str else '(not set)')}")
    print(f"Board:   {(' '.join(board_str) if board_str else '(none)')}")
    print("-----------\n")

def main():
    print("\n=== Poker MC Session (Win% + Kelly%) ===")
    print("Type /help for commands. Set your hand once, then /add cards as the board appears.\n")

    players = DEFAULT_PLAYERS
    trials  = DEFAULT_TRIALS
    my_hand_str = []
    board_str   = []

    # Require a hand once at start:
    while not my_hand_str:
        raw = input("Enter your hand (2 cards, e.g., 'As Kh'): ").strip()
        if raw.lower() in ("/help", "-help"):
            print(HELP_TEXT); continue
        try:
            cards = tokenize_cards(raw)
            if len(cards) != 2:
                print("Please enter exactly 2 cards.")
                continue
            if len(set(cards)) != 2:
                print("Duplicate cards detected.")
                continue
            my_hand_str = cards
        except Exception as e:
            print(f"Parse error: {e}")

    # Main interactive loop
    while True:
        raw = input("Command or board cards to add (/help): ").strip()
        low = raw.lower()

        if low in ("/quit", "/q", "quit", "exit"):
            print("Good luck!"); break

        if low in ("/help", "-help"):
            print(HELP_TEXT); continue

        if low.startswith("/status"):
            show_status(players, trials, my_hand_str, board_str); continue

        if low.startswith("/players"):
            parts = low.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("Usage: /players N  (2..10)")
                continue
            n = int(parts[1]); 
            if n < 2 or n > 10:
                print("Players must be 2..10."); continue
            players = n
            print(f"Players set to {players}.")
            # run immediately with current board
        elif low.startswith("/trials"):
            parts = low.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("Usage: /trials N  (>=1000)")
                continue
            t = int(parts[1])
            if t < 1000:
                print("Trials must be >= 1000."); continue
            trials = t
            print(f"Trials set to {trials:,}.")
        elif low.startswith("/hand"):
            # allow resetting hand
            rest = raw[len("/hand"):].strip()
            try:
                cards = tokenize_cards(rest)
                if len(cards) != 2:
                    print("Usage: /hand <exactly 2 cards>")
                    continue
                if len(set(cards)) != 2 or any(c in board_str for c in cards):
                    print("Cards must be unique and not overlap with current board.")
                    continue
                my_hand_str = cards
                print(f"Hand set to: {' '.join(my_hand_str)}")
                board_str = []  # usually start a new board with a new hand
                print("Board cleared for new hand.")
            except Exception as e:
                print(f"Parse error: {e}")
                continue
        elif low in ("/clear",):
            board_str = []
            print("Board cleared.")
        elif low in ("/new",):
            board_str = []
            print("Board cleared. Enter /hand <cards> to change your hand (or keep current).")
        elif low.startswith("/add"):
            rest = raw[len("/add"):].strip()
            try:
                add_cards = tokenize_cards(rest)
                if not add_cards:
                    print("Usage: /add <one or more cards>")
                    continue
                # Validate uniqueness and capacity
                combined = my_hand_str + board_str + add_cards
                if len(set(combined)) != len(combined):
                    print("Duplicate or overlapping cards detected.")
                    continue
                if len(board_str) + len(add_cards) > 5:
                    print("Board cannot exceed 5 cards total.")
                    continue
                board_str += add_cards
            except Exception as e:
                print(f"Parse error: {e}")
                continue
        else:
            # Treat raw input as board cards to add (convenience)
            try:
                add_cards = tokenize_cards(raw)
                if not add_cards:
                    print("Enter cards or a command. Type /help for options.")
                    continue
                combined = my_hand_str + board_str + add_cards
                if len(set(combined)) != len(combined):
                    print("Duplicate or overlapping cards detected.")
                    continue
                if len(board_str) + len(add_cards) > 5:
                    print("Board cannot exceed 5 cards total.")
                    continue
                board_str += add_cards
            except Exception as e:
                print(f"Parse error: {e}")
                continue

        # ---- Run simulation after any valid change or setting tweak ----
        try:
            my_cards = to_treys(my_hand_str)
            board_cards = to_treys(board_str)
            win_p, tie_p, equity = simulate(players, my_cards, board_cards, trials, seed=RNG_SEED)
            f = kelly_even_money(equity)  # % of bankroll
            # CI
            var = equity * (1 - equity) / max(1, trials)
            moe = 1.96 * math.sqrt(var)
            print("\n--- Results ---")
            print(f"Players: {players}   Trials: {trials:,}")
            print(f"Hand: {' '.join(my_hand_str)}   Board: {(' '.join(board_str) if board_str else '(none)')}")
            print(f"Win%: {win_p*100:.2f}   Tie%: {tie_p*100:.2f}   Equity: {equity*100:.2f}%")
            print(f"Kelly (even-money): {f*100:.2f}% of bankroll")
            print(f"95% CI (equity): ±{moe*100:.2f}%")
            print("----------------\n")
        except Exception as e:
            print(f"Simulation error: {e}")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
