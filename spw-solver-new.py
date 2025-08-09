import os
import random
import time
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
import math

# ----------------- CONFIG (63-letter) -----------------
LETTER_COUNT = 63
GRID_ROWS = 7
GRID_COLS = 10
ANCHOR_ROWS_PER_SET = 4 #GRID_ROWS // 2
NUM_PROCESSES = 1 #GRID_ROWS
MAX_GENERATION_CYCLES = 100000 # Maximum cycles to try generating anchors
CYCLE_REPORT_INTERVAL = 1
BACKTRACK_REPORT_INTERVAL = 1 #100
MAX_ROW_SEGMENT_PATTERNS = 200           # Maximum hash patterns to try per row
MAX_WORDS_PER_SEGMENT = 1000            # Maximum words to try per segment
MAX_HASH_PLACEMENT_ATTEMPTS = 500         # Number of hash placement retries per anchor set
MAX_FAILED_STATE_CACHE = 100000  # Maximum number of failed states to remember
MAX_ATTEMPTS_BEFORE_RESTART = 5000  # Maximum backtracking attempts before restart

# ----------------- CONFIG (21-letter) -----------------
# LETTER_COUNT = 21
# GRID_ROWS = 3
# GRID_COLS = 8
# ANCHOR_ROWS_PER_SET = 2 #GRID_ROWS // 2
# NUM_PROCESSES = 8 #GRID_ROWS
# MAX_GENERATION_CYCLES = 10000 # Maximum cycles to try generating anchors
# CYCLE_REPORT_INTERVAL = 1000
# BACKTRACK_REPORT_INTERVAL = 10000 #100
# MAX_ROW_SEGMENT_PATTERNS = 10           # Maximum hash patterns to try per row
# MAX_WORDS_PER_SEGMENT = 1000            # Maximum words to try per segment
# MAX_HASH_PLACEMENT_ATTEMPTS = 100         # Number of hash placement retries per anchor set
# MAX_FAILED_STATE_CACHE = 100000  # Maximum number of failed states to remember

# ---------- OTHER GLOBALS ----------
BITS_PER_CHAR = 5
WILDCARD = (1 << BITS_PER_CHAR) - 1

# ---------- MASK HELPERS ----------
def word_to_mask(word):
    mask = 0
    for i, ch in enumerate(word):
        val = ord(ch) - 65
        mask |= val << (i * BITS_PER_CHAR)
    return mask

def build_bitmask_dict_by_length(words):
    bitmask_dict_by_length = defaultdict(set)
    for w in words:
        L = len(w)
        bitmask_dict_by_length[L].add(word_to_mask(w))
    return bitmask_dict_by_length

def pattern_to_mask(segment):
    mask = 0
    for i, ch in enumerate(segment):
        val = WILDCARD if ch == '.' else (ord(ch) - 65)
        mask |= val << (i * BITS_PER_CHAR)
    return mask

def mask_matches_pattern(word_mask, pattern_mask, length, wildcard=WILDCARD):
    for i in range(length):
        shift = i * BITS_PER_CHAR
        pattern_bits = (pattern_mask >> shift) & wildcard
        if pattern_bits != wildcard:
            word_bits = (word_mask >> shift) & wildcard
            if pattern_bits != word_bits:
                return False
    return True

def fits_pattern(word, pattern):
    if len(word) != len(pattern):
        return False
    word_mask = word_to_mask(word)
    pattern_mask = pattern_to_mask(pattern)
    return mask_matches_pattern(word_mask, pattern_mask, len(word))

def matches_any(pattern_mask, bitmask_dict_by_length, length):
    candidates = bitmask_dict_by_length.get(length, set())
    for m in candidates:
        if mask_matches_pattern(m, pattern_mask, length):
            return True
    return False


# ---------- SEGMENT HELPERS ----------
def get_segments(line):
    """Return a list of (start_index, segment_chars) for all non-# segments in a line."""
    segments = []
    current = []
    for idx, ch in enumerate(line):
        if ch == '#':
            if current:
                segments.append((idx - len(current), current[:]))
                current = []
        else:
            current.append(ch)
    if current:
        segments.append((len(line) - len(current), current[:]))
    return segments

def is_valid_grid_segments(grid, dict_by_length):
    """Check all segments in all rows and columns for validity (filled segments must be valid words)."""
    for r in range(grid.rows):
        row = grid.cells[r]
        for start, segment in get_segments(row):
            seg_str = ''.join(segment)
            if '.' not in seg_str and len(seg_str) > 1:
                if seg_str not in dict_by_length[len(seg_str)]:
                    return False
    for c in range(grid.cols):
        col = [grid.cells[r][c] for r in range(grid.rows)]
        for start, segment in get_segments(col):
            seg_str = ''.join(segment)
            if '.' not in seg_str and len(seg_str) > 1:
                if seg_str not in dict_by_length[len(seg_str)]:
                    return False
    return True



# ---------- GRID CLASS ----------
class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cells = [['.' for _ in range(cols)] for _ in range(rows)]

    def copy(self):
        new_g = Grid(self.rows, self.cols)
        for r in range(self.rows):
            new_g.cells[r] = self.cells[r][:]
        return new_g

    def place(self, word, r, c, direction):
        if direction == 'H':
            for i, ch in enumerate(word):
                self.cells[r][c + i] = ch
        else:
            for i, ch in enumerate(word):
                self.cells[r + i][c] = ch

    def insert_gap(self, r, pos):
        self.cells[r][pos] = '#'

    def can_place_segment(self, word, r, c):
        if c + len(word) > self.cols:
            return False
        for i, ch in enumerate(word):
            if self.cells[r][c + i] not in ('.', ch):
                return False
        return True

    def coverage(self):
        filled = sum(ch != '.' for row in self.cells for ch in row)
        return filled / (self.rows * self.cols)

    def validate_vertical_segments(self, bitmask_dict_by_length):
        for col in range(self.cols):
            col_cells = [self.cells[r][col] for r in range(self.rows)]
            for start, segment in get_segments(col_cells):
                seg_str = ''.join(segment)
                if len(seg_str.strip('.')) <= 1:
                    continue
                seg_mask = pattern_to_mask(seg_str)
                if not matches_any(seg_mask, bitmask_dict_by_length, len(seg_str)):
                    return False
        return True

    def as_tuple(self):
        return tuple(''.join(row) for row in self.cells)

    def show(self):
        return "\n".join("".join(row) for row in self.cells)

# ---------- DICTIONARY ----------
def preprocess_dictionary(path, available_letters):
    with open(path, "r") as f:
        words = [w.strip().upper() for w in f if w.strip().isalpha() and len(w.strip()) <= GRID_COLS]
    dict_by_length = defaultdict(list)
    for w in words:
        word_count = Counter(w)
        if all(available_letters[ch] >= cnt for ch, cnt in word_count.items()):
            dict_by_length[len(w)].append(w)
    for length in dict_by_length:
        dict_by_length[length].sort()
    # Debug: print number and sample of 10-letter words
    if 10 in dict_by_length:
        print(f"[DEBUG] Loaded {len(dict_by_length[10])} 10-letter words.")
        print(f"[DEBUG] Sample 10-letter words: {dict_by_length[10][:5]}")
    mask_dict = build_bitmask_dict_by_length([w for words in dict_by_length.values() for w in words])
    return dict_by_length, mask_dict


# ---------- LINE CONTENT GENERATOR (Generalized) ----------

# ---------- UNIFIED LINE FILL GENERATOR ----------
def find_single_line_fill(line, available, dict_by_length, used_words, max_hashes, mask_dict):
    """
    Try to find a single valid fill for a line (row or column), given available letters,
    used words, and remaining hashes. Returns (fill_string, new_available, new_used_words, nh) or None.
    """
    line_length = len(line)
    # Try fills with the minimum number of hashes first (for each nh, try all possible hash positions in random order)
    for nh in range(0, max_hashes + 1):
        hash_positions_list = list(combinations(range(line_length), nh))
        random.shuffle(hash_positions_list)
        for hash_positions in hash_positions_list:
            fill_pattern = list(line)
            for pos in hash_positions:
                fill_pattern[pos] = '#'
            if all(ch == '#' for ch in fill_pattern):
                continue
            segments = get_segments(fill_pattern)
            temp_available = available.copy()
            temp_used_words = used_words.copy()
            fill_string = fill_pattern[:]
            valid = True
            for start, seg in segments:
                seg_len = len(seg)
                if seg_len <= 1:
                    continue
                candidates = dict_by_length.get(seg_len, [])[:MAX_WORDS_PER_SEGMENT]
                random.shuffle(candidates)
                found = False
                for word in candidates:
                    if word in temp_used_words:
                        continue
                    if not can_use_word(word, temp_available):
                        continue
                    for i, ch in enumerate(word):
                        fill_string[start + i] = ch
                    use_word_on_segment(word, seg, temp_available)
                    temp_used_words.add(word)
                    found = True
                    break
                if not found:
                    valid = False
                    break
            if valid:
                return ''.join(fill_string), temp_available, temp_used_words, nh
    return None


# ---------- UNIFIED LINE FILL AND SELECTION ----------
def fill_grid_line_once(grid, available, used_words, is_row, idx, dict_by_length, mask_dict, max_hashes):
    if is_row:
        line = grid.cells[idx][:]
    else:
        line = [grid.cells[r][idx] for r in range(grid.rows)]
    import copy
    # Try all fills with increasing number of hashes, backtracking if needed
    for nh in range(0, max_hashes + 1):
        hash_positions_list = list(combinations(range(len(line)), nh))
        random.shuffle(hash_positions_list)
        for hash_positions in hash_positions_list:
            fill_pattern = list(line)
            for pos in hash_positions:
                fill_pattern[pos] = '#'
            if all(ch == '#' for ch in fill_pattern):
                continue
            segments = get_segments(fill_pattern)
            temp_available = available.copy()
            temp_used_words = used_words.copy()
            fill_string = fill_pattern[:]
            valid = True
            for start, seg in segments:
                seg_len = len(seg)
                if seg_len <= 1:
                    continue
                candidates = dict_by_length.get(seg_len, [])[:MAX_WORDS_PER_SEGMENT]
                random.shuffle(candidates)
                found = False
                for word in candidates:
                    if word in temp_used_words:
                        continue
                    if not can_use_word(word, temp_available):
                        continue
                    for i, ch in enumerate(word):
                        fill_string[start + i] = ch
                    use_word_on_segment(word, seg, temp_available)
                    temp_used_words.add(word)
                    found = True
                    break
                if not found:
                    valid = False
                    break
            if valid:
                new_grid = copy.deepcopy(grid)
                if is_row:
                    for i in range(len(fill_string)):
                        new_grid.cells[idx][i] = fill_string[i]
                else:
                    for i in range(len(fill_string)):
                        new_grid.cells[i][idx] = fill_string[i]
                print(f"[DEBUG] fill_grid_line_once: filled {'row' if is_row else 'col'} {idx} (hashes used: {nh})\n{new_grid.show()}", flush=True)
                return new_grid, temp_available.copy(), temp_used_words.copy(), nh
    print(f"[DEBUG] fill_grid_line_once: no valid fill for {'row' if is_row else 'col'} {idx}", flush=True)
    return None

def select_line(grid, available, dict_by_length, mask_dict):
    # MRV: select the line (row or col) with the fewest candidates
    best_candidates = None
    is_row_best = True
    idx_best = -1
    for r in range(grid.rows):
        row = grid.cells[r]
        row_str = ''.join(row)
        print(f"[DEBUG] select_line: row {r} contents: {row_str}", flush=True)
        if '.' not in row:
            print(f"[DEBUG] select_line: skipping row {r} (already filled)", flush=True)
            continue
        print(f"[DEBUG] select_line: considering row {r}", flush=True)
        pattern = row_str
        pattern_mask = pattern_to_mask(pattern)
        candidates = [w for w in dict_by_length[len(pattern)] if mask_matches_pattern(word_to_mask(w), pattern_mask, len(pattern)) and can_use_word(w, available)]
        if r == 0 and all(ch == '.' for ch in row):
            print(f"[DEBUG] Candidates for row 0: {candidates[:10]} (total: {len(candidates)})")
        if best_candidates is None or len(candidates) < len(best_candidates):
            best_candidates = candidates
            is_row_best = True
            idx_best = r
    for c in range(grid.cols):
        col = [grid.cells[r][c] for r in range(grid.rows)]
        col_str = ''.join(col)
        print(f"[DEBUG] select_line: col {c} contents: {col_str}", flush=True)
        if '.' not in col:
            print(f"[DEBUG] select_line: skipping col {c} (already filled)", flush=True)
            continue
        print(f"[DEBUG] select_line: considering col {c}", flush=True)
        pattern = col_str
        pattern_mask = pattern_to_mask(pattern)
        candidates = [w for w in dict_by_length[len(pattern)] if mask_matches_pattern(word_to_mask(w), pattern_mask, len(pattern)) and can_use_word(w, available)]
        if best_candidates is None or len(candidates) < len(best_candidates):
            best_candidates = candidates
            is_row_best = False
            idx_best = c
    if best_candidates is None:
        print("[DEBUG] select_line: no line selected (no candidates)", flush=True)
        return None, None
    print(f"[DEBUG] select_line: next is {'row' if is_row_best else 'col'} {idx_best} with {len(best_candidates)} candidates", flush=True)
    if is_row_best:
        print(f"[DEBUG] select_line: SELECTED row {idx_best}: {''.join(grid.cells[idx_best])}", flush=True)
    else:
        col = ''.join(grid.cells[r][idx_best] for r in range(grid.rows))
        print(f"[DEBUG] select_line: SELECTED col {idx_best}: {col}", flush=True)
    return is_row_best, idx_best


# ---------- MAIN SOLVER (BACKTRACKING) ----------
def progressive_and_backtracking_fill(grid, available, used_words, dict_by_length, mask_dict, threshold_lines):
    filled_lines = 0
    stack = [(grid, available, used_words.copy(), filled_lines)]
    allowed_hashes = (grid.rows * grid.cols) - LETTER_COUNT
    import copy
    while stack:
        cur_grid, cur_available, cur_used_words, cur_filled_lines = stack.pop()
        print("[DEBUG] v4 progressive_and_backtracking_fill: (after pop) current grid state:")
        print(cur_grid.show())
        for i, row in enumerate(cur_grid.cells):
            if '.' in row:
                print(f"[DEBUG] (after pop) Row {i} has dots: {''.join(row)}")
        for j in range(cur_grid.cols):
            col = ''.join(cur_grid.cells[i][j] for i in range(cur_grid.rows))
            if '.' in col:
                print(f"[DEBUG] (after pop) Col {j} has dots: {col}")
        total_hashes = sum(row.count('#') for row in cur_grid.cells)
        if total_hashes > allowed_hashes:
            continue
        all_hash_row = any(all(ch == '#' for ch in row) for row in cur_grid.cells)
        all_hash_col = any(all(cur_grid.cells[r][c] == '#' for r in range(cur_grid.rows)) for c in range(cur_grid.cols))
        if all_hash_row or all_hash_col:
            continue
        if all('.' not in row for row in cur_grid.cells):
            if validate_full_grid(cur_grid, dict_by_length) and is_connected(cur_grid):
                print("✅ Solution Found:")
                print(cur_grid.show())
                return cur_grid
            continue
        is_row, idx = select_line(cur_grid, cur_available, dict_by_length, mask_dict)
        if is_row is not None:
            if is_row:
                print(f"[DEBUG] (after pop) select_line chose row {idx}: {''.join(cur_grid.cells[idx])}", flush=True)
            else:
                col_str = ''.join(cur_grid.cells[r][idx] for r in range(cur_grid.rows))
                print(f"[DEBUG] (after pop) select_line chose col {idx}: {col_str}", flush=True)
        else:
            print(f"[DEBUG] (after pop) select_line found no line to fill", flush=True)
        if is_row is None:
            continue
        if is_row:
            print(f"[DEBUG] about to fill row {idx}: {''.join(cur_grid.cells[idx])}", flush=True)
        else:
            col_str = ''.join(cur_grid.cells[r][idx] for r in range(cur_grid.rows))
            print(f"[DEBUG] about to fill col {idx}: {col_str}", flush=True)
        max_hashes = sum(1 for ch in (cur_grid.cells[idx] if is_row else [cur_grid.cells[r][idx] for r in range(cur_grid.rows)]) if ch == '.')
        fill_result = fill_grid_line_once(cur_grid, cur_available, cur_used_words, is_row, idx, dict_by_length, mask_dict, max_hashes)
        if fill_result is not None:
            new_grid, new_available, new_used_words, nh = fill_result
            filled_line = new_grid.cells[idx] if is_row else [new_grid.cells[r][idx] for r in range(new_grid.rows)]
            if '.' in filled_line:
                print(f"[DEBUG] skipping push: line {'row' if is_row else 'col'} {idx} not fully filled after fill: {''.join(filled_line)}", flush=True)
                continue
            print(f"[DEBUG] progressive_and_backtracking_fill: pushing state with next line {'row' if is_row else 'col'} {idx}\n{new_grid.show()}", flush=True)
            stack.append((new_grid, new_available, new_used_words.copy(), cur_filled_lines + 1))
        else:
            print(f"[DEBUG] No valid fill found for {'row' if is_row else 'col'} {idx}, backtracking.", flush=True)
    print("❌ No solution found.")
    return None

# ---------- SOLVER UTILITIES ----------
def find_next_empty(grid):
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.cells[r][c] == '.':
                return r, c
    return None

def get_horizontal_segment(grid, r, c):
    row = grid.cells[r]
    start = c
    while start > 0 and row[start - 1] not in ('#',):
        start -= 1
    end = c
    while end < grid.cols - 1 and row[end + 1] not in ('#',):
        end += 1
    return start, end

def get_vertical_segment(grid, r, c):
    start = r
    while start > 0 and grid.cells[start - 1][c] not in ('#',):
        start -= 1
    end = r
    while end < grid.rows - 1 and grid.cells[end + 1][c] not in ('#',):
        end += 1
    return start, end

def validate_full_grid(grid, dict_by_length):
    for r in range(grid.rows):
        row = ''.join(grid.cells[r])
        for seg in row.replace('#', ' ').split():
            if len(seg) > 1 and seg not in dict_by_length[len(seg)]:
                return False
    for c in range(grid.cols):
        col = ''.join(grid.cells[r][c] for r in range(grid.rows))
        for seg in col.replace('#', ' ').split():
            if len(seg) > 1 and seg not in dict_by_length[len(seg)]:
                return False
    return True

def can_use_word(word, available):
    word_count = Counter(word)
    for ch, cnt in word_count.items():
        if available[ch] < cnt:
            return False
    return True

def use_word_on_segment(word, segment, available):
    """
    Decrement only the letters that are actually placed (i.e., where segment has '.').
    """
    for i, ch in enumerate(word):
        if segment[i] == '.':
            available[ch] -= 1

def is_connected(grid):
    visited = set()
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.cells[r][c] != '#':
                start = (r, c)
                break
        else:
            continue
        break
    else:
        return True

    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                if grid.cells[nr][nc] != '#' and (nr, nc) not in visited:
                    stack.append((nr, nc))
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.cells[r][c] != '#' and (r, c) not in visited:
                return False
    return True

def check_endgame(grid, dict_by_length):
    is_conn = is_connected(grid)
    if not is_conn:
        print("⚠️ Disconnected grid found, skipping.")
        print(grid.show())
        return False
    has_dot = any('.' in row for row in grid.cells)
    if has_dot:
        print("⚠️ Grid not fully filled, skipping.")
        print(grid.show())
        raise ValueError("Grid not fully filled, but no more words available.")
        # return False
    valid = validate_full_grid(grid, dict_by_length)
    if not valid:
        print("⚠️ Invalid grid found, skipping.")
        print(grid.show())
        return False
    return True

# ---------- BACKTRACKING ----------


def solve_spaceword(letters, dictionary_path="wCSW.txt"):
    if len(letters) != LETTER_COUNT:
        raise ValueError(f"Expected {LETTER_COUNT} letters, got {len(letters)}. Please adjust the configuration.")
    
    if GRID_ROWS > GRID_COLS:
        raise ValueError("GRID_ROWS must be <= GRID_COLS. Tall grids are not supported and can be obtained by transposing the grid. Please adjust the configuration.")

    num_hashes = (GRID_ROWS * GRID_COLS) - LETTER_COUNT
    if num_hashes < 0:
        raise ValueError("Not enough letters to fill the grid, please provide more letters or reduce the grid size.")

    print("Loading dictionary...")
    puzzle_letter_counter = Counter(letters)
    dict_by_length, mask_dict = preprocess_dictionary(dictionary_path, puzzle_letter_counter)
    # Debug: print available letter counts
    print(f"[DEBUG] Available letter counts: {dict(puzzle_letter_counter)}")
    for length in dict_by_length:
        print(f"Dictionary length {length}: {len(dict_by_length[length])} words")

    print("\nStarting unified fill and backtracking...")
    start_time = time.time()
    grid = Grid(GRID_ROWS, GRID_COLS)
    used_words = set()
    solution = progressive_and_backtracking_fill(grid, puzzle_letter_counter.copy(), used_words, dict_by_length, mask_dict, threshold_lines=0)
    elapsed = time.time() - start_time
    if solution:
        print(f"Total time: {elapsed:.2f}s", flush=True)
    else:
        print("❌ No solution found yet. Consider adjusting the parameters.")

if __name__ == "__main__":
    puzzle_letters = "AAAAABBBCCDDDEEEEEEEEFFFGGGHHHIILMMMNNOOOOPPPQRSSSSTTTTUUUVXYYZ"
    # puzzle_letters = "ADEEEEEEGHIKNOQRRTUWY"
    solve_spaceword(puzzle_letters, "wCSW.txt")