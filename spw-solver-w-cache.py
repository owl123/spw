import os
import random
import time
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from multiprocessing import Manager

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

# ---------- GRID ----------
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
    mask_dict = build_bitmask_dict_by_length([w for words in dict_by_length.values() for w in words])
    return dict_by_length, mask_dict

# ---------- RANDOM HASH PLACEMENT on GRID ----------
def grid_hash_placement(grid, num_hashes, dict_by_length, max_hashes, mask_dict):
    """
    Place hashes randomly in the grid, ensuring all constraints are met.
    """
    def get_empty_cells(grid):
        return [(r, c) for r in range(grid.rows) for c in range(grid.cols) if grid.cells[r][c] == '.']
    def place_hash(grid, pos):
        r, c = pos
        grid.cells[r][c] = '#'
    def remove_hash(grid, pos):
        r, c = pos
        grid.cells[r][c] = '.'

    def is_valid_hash_placement(grid):
        for c in range(grid.cols):
            col = [grid.cells[r][c] for r in range(grid.rows)]
            for start, segment in get_segments(col):
                seg_str = ''.join(segment)
                if len(segment) <= 1:
                    continue
                if '.' not in seg_str:
                    # Fully filled: must be a valid word
                    if seg_str not in dict_by_length[len(seg_str)]:
                        return False
                elif all(ch == '.' for ch in segment):
                    # All empty: must be fillable
                    if len(dict_by_length[len(segment)]) == 0:
                        return False
                else:
                    # Partially filled: must match at least one word
                    pattern_mask = pattern_to_mask(seg_str)
                    if not matches_any(pattern_mask, mask_dict, len(seg_str)):
                        return False
        return True

    def place_hashes_recursive(grid, num_hashes_placed):
        if num_hashes_placed == max_hashes:
            return is_valid_hash_placement(grid)

        empty_cells = get_empty_cells(grid)
        if not empty_cells:
            return False

        random.shuffle(empty_cells)  # Pure random order

        for pos in empty_cells:
            place_hash(grid, pos)
            if not is_connected(grid):
                remove_hash(grid, pos)
                continue
            if not is_valid_hash_placement(grid):
                remove_hash(grid, pos)
                continue
            if place_hashes_recursive(grid, num_hashes_placed + 1):
                return True
            remove_hash(grid, pos)
        return False

    return place_hashes_recursive(grid, num_hashes)

# ---------- LINE CONTENT GENERATOR (Generalized) ----------
def generate_line_content(line_idx, is_row, available, shuffled_dicts, max_hashes, used_words, grid, mask_dict):
    line_length = grid.cols if is_row else grid.rows

    hash_patterns = []
    for nh in range(0, max_hashes + 1):
        patterns = list(combinations(range(line_length), nh))
        random.shuffle(patterns)
        hash_patterns.extend(patterns)

    for combo in hash_patterns[:MAX_ROW_SEGMENT_PATTERNS]:
        fill_pattern = ['.'] * line_length
        for pos in combo:
            fill_pattern[pos] = '#'
        segments = get_segments(fill_pattern)
        temp_available = available.copy()
        temp_used_words = used_words.copy()
        fill_string = fill_pattern[:]
        valid = True
        for start, seg in segments:
            seg_len = len(seg)
            found = False
            candidates = shuffled_dicts.get(seg_len, [])[:MAX_WORDS_PER_SEGMENT]
            if not candidates:
                valid = False
                break
            tries = 0
            while tries < min(len(candidates), MAX_WORDS_PER_SEGMENT):
                word = random.choice(candidates)
                if word not in temp_used_words and can_use_word(word, temp_available):
                    for i, ch in enumerate(word):
                        fill_string[start + i] = ch
                    use_word_on_segment(word, seg, temp_available)
                    temp_used_words.add(word)
                    found = True
                    break
                tries += 1
            if not found:
                valid = False
                break
        if not valid:
            continue

        temp_grid = grid.copy()
        if is_row:
            temp_grid.cells[line_idx] = fill_string[:]
        else:
            for i in range(len(fill_string)):
                temp_grid.cells[i][line_idx] = fill_string[i]
        # Validate the orthogonal direction
        if is_row:
            if temp_grid.validate_vertical_segments(mask_dict):
                return ''.join(fill_string), temp_grid, temp_available, temp_used_words
        else:
            # Validate horizontal segments
            for r in range(temp_grid.rows):
                row = temp_grid.cells[r]
                for start, segment in get_segments(row):
                    seg_str = ''.join(segment)
                    if len(seg_str.strip('.')) <= 1:
                        continue
                    seg_mask = pattern_to_mask(seg_str)
                    if not matches_any(seg_mask, mask_dict, len(seg_str)):
                        break
                else:
                    continue
                break
            else:
                return ''.join(fill_string), temp_grid, temp_available, temp_used_words
    return None, None, None, None

# ---------- ANCHOR GENERATION (Alternating Rows/Cols) ----------
def generate_anchor_sets(seed, puzzle_letter_counter, mask_dict, dict_by_length, seen):
    num_hashes = (GRID_ROWS * GRID_COLS) - LETTER_COUNT
    rows = list(range(GRID_ROWS))
    cols = list(range(GRID_COLS))
    cycle_count = 0

    # Decide anchor order: alternate rows and columns if aspect ratio < 2
    aspect_ratio = GRID_COLS / GRID_ROWS
    anchor_order = []
    if aspect_ratio < 2:
        # Alternate between rows and columns
        min_dim = min(GRID_ROWS, GRID_COLS)
        row_indices = list(range(GRID_ROWS))
        col_indices = list(range(GRID_COLS))
        random.shuffle(row_indices)
        random.shuffle(col_indices)
        for i in range(ANCHOR_ROWS_PER_SET):
            if i % 2 == 0 and row_indices:
                anchor_order.append(('row', row_indices.pop()))
            elif col_indices:
                anchor_order.append(('col', col_indices.pop()))
    else:
        # Use only rows (original logic)
        seed_row = seed % GRID_ROWS
        other_rows = [r for r in rows if r != seed_row]
        anchor_rows = [seed_row] + random.sample(other_rows, ANCHOR_ROWS_PER_SET - 1)
        anchor_order = [('row', r) for r in anchor_rows]

    while cycle_count < MAX_GENERATION_CYCLES:
        cycle_count += 1
        if cycle_count % CYCLE_REPORT_INTERVAL == 0:
            print(f"[Seed {seed}] Cycle {cycle_count}/{MAX_GENERATION_CYCLES} - {len(seen)} unique grids found")
        grid = Grid(GRID_ROWS, GRID_COLS)
        available = puzzle_letter_counter.copy()
        used_words = set()
        anchors = []
        hashes_used = 0
        valid = True

        for anchor_type, idx in anchor_order:
            result = generate_line_content(
                idx,
                is_row=(anchor_type == 'row'),
                available=available,
                shuffled_dicts=dict_by_length,
                max_hashes=num_hashes - hashes_used,
                used_words=used_words,
                grid=grid,
                mask_dict=mask_dict
            )
            fill_string, temp_grid, temp_available, temp_used_words = result
            if fill_string is not None:
                grid = temp_grid
                available = temp_available
                used_words = temp_used_words
                anchors.append((fill_string, anchor_type, idx))
                hashes_used += fill_string.count('#')
            else:
                valid = False
                break

        if not valid or len(anchors) < ANCHOR_ROWS_PER_SET:
            continue

        empty_cells = sum(row.count('.') for row in grid.cells)
        remaining_hashes = num_hashes - hashes_used
        if empty_cells < remaining_hashes:
            continue

        for hash_attempt in range(MAX_HASH_PLACEMENT_ATTEMPTS):
            grid_with_hashes = grid.copy()
            if not grid_hash_placement(grid_with_hashes, 0, dict_by_length, remaining_hashes, mask_dict):
                continue

            grid_tuple = grid_with_hashes.as_tuple()
            if grid_tuple in seen:
                continue
            seen.append(grid_tuple)
            print(grid_with_hashes.show())

            if backtrack_fill(seed, grid_with_hashes, dict_by_length, mask_dict, available):
                print(f"[Seed {seed}] ✅ Solution Found:")
                print(grid_with_hashes.show())
                print(f"Anchors: {anchors}")
                return [(grid_with_hashes, anchors)]
    return []

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
def find_mrv_segment(grid, available, dict_by_length, mask_dict):
    """
    Find the segment (horizontal or vertical) with the fewest candidate words (MRV heuristic).
    Returns a tuple: (direction, row/col, start, candidates, pattern)
    """
    min_candidates = float('inf')
    best = None

    # Check all horizontal segments
    for r in range(grid.rows):
        row = grid.cells[r]
        for start, segment in get_segments(row):
            seg_str = ''.join(segment)
            if len(seg_str) > 1 and '.' in seg_str and '#' not in seg_str:
                pattern_mask = pattern_to_mask(seg_str)
                candidates = [
                    w for w in dict_by_length[len(seg_str)]
                    if mask_matches_pattern(word_to_mask(w), pattern_mask, len(seg_str)) and can_use_word(w, available)
                ]
                if 0 < len(candidates) < min_candidates:
                    min_candidates = len(candidates)
                    best = ('H', r, start, candidates, seg_str)

    # Check all vertical segments
    for c in range(grid.cols):
        col = [grid.cells[r][c] for r in range(grid.rows)]
        for start, segment in get_segments(col):
            seg_str = ''.join(segment)
            if len(seg_str) > 1 and '.' in seg_str and '#' not in seg_str:
                pattern_mask = pattern_to_mask(seg_str)
                candidates = [
                    w for w in dict_by_length[len(seg_str)]
                    if mask_matches_pattern(word_to_mask(w), pattern_mask, len(seg_str)) and can_use_word(w, available)
                ]
                if 0 < len(candidates) < min_candidates:
                    min_candidates = len(candidates)
                    best = ('V', c, start, candidates, seg_str)

    return best  # or None if no fillable segment found

def backtrack_fill(seed, grid, dict_by_length, mask_dict, available, attempt=[0], failed_states=None):
    if failed_states is None:
        failed_states = OrderedDict()

    state_key = (grid.as_tuple(), tuple(sorted(available.items())))
    if state_key in failed_states:
        return False

    if not any(available.values()):
        print("DEBUG: All letters used up, but grid state is:")
        print(grid.show())
        print("Available:", available)
        print("Dots left:", sum(row.count('.') for row in grid.cells))
        return check_endgame(grid, dict_by_length)

    empty = find_next_empty(grid)
    if not empty:
        return check_endgame(grid, dict_by_length)

    r, c = empty
    attempt[0] += 1
    if attempt[0] % BACKTRACK_REPORT_INTERVAL == 0:
        print(f"[Seed {seed}] Backtracking attempts: {attempt[0]} (coverage: {grid.coverage():.2f})")
        print("Current grid state:")
        print(grid.show())
        print("-" * 20)

    mrv = find_mrv_segment(grid, available, dict_by_length, mask_dict)
    if not mrv:
        failed_states[state_key] = None
        if len(failed_states) > MAX_FAILED_STATE_CACHE:
            failed_states.popitem(last=False)
        return False  # No fillable segment found

    direction, idx, start, candidates, pattern = mrv
    for word in candidates:
        new_grid = grid.copy()
        new_available = available.copy()
        if direction == 'H':
            segment = [grid.cells[idx][start + i] for i in range(len(word))]
            new_grid.place(word, idx, start, 'H')
        else:
            segment = [grid.cells[start + i][idx] for i in range(len(word))]
            new_grid.place(word, start, idx, 'V')
       
        if is_valid_grid_segments(new_grid, dict_by_length):
            use_word_on_segment(word, segment, new_available)
            if backtrack_fill(seed, new_grid, dict_by_length, mask_dict, new_available, attempt, failed_states):
                grid.cells = new_grid.cells
                return True

    failed_states[state_key] = None
    if len(failed_states) > MAX_FAILED_STATE_CACHE:
        failed_states.popitem(last=False)
    return False

# ---------- SOLVER ENTRY POINT WITH SUPPORT FOR MULTI-PROCESSING ----------
def run_solver(args):
    return generate_anchor_sets(*args)

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
    for length in dict_by_length:
        print(f"Dictionary length {length}: {len(dict_by_length[length])} words")

    print("\nStarting anchor generation and backtracking...")
    manager = Manager()
    seen = manager.list()  # Shared set to avoid duplicate grids
    start_time = time.time()
    seeds = list(range(NUM_PROCESSES))
    seed_args = [
        (seed, puzzle_letter_counter, mask_dict, dict_by_length, seen)
        for seed in seeds
    ]

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        found = False
        try:
            for res in executor.map(run_solver, seed_args):
                if res:
                    # Each res is a list of (working_grid, anchors) where a solution was found
                    for (solution_grid, anchors) in res:
                        print("✅ Solution Found:")
                        print(solution_grid.show())
                        print(f"Anchors: {anchors}")
                        elapsed = time.time() - start_time
                        print(f"Total time: {elapsed:.2f}s", flush=True)
                        found = True
                        break  # Break inner loop after first solution
                if found:
                    break  # Break outer loop after first solution
            if not found:
                print("❌ No solution found yet. Consider adjusting the parameters.")
        except Exception as e:
            print("Fatal error in worker process:", e)
            executor.shutdown(wait=False, cancel_futures=True)
            os._exit(1)

if __name__ == "__main__":
    puzzle_letters = "AAAAABBBCCDDDEEEEEEEEFFFGGGHHHIILMMMNNOOOOPPPQRSSSSTTTTUUUVXYYZ"
    # puzzle_letters = "ADEEEEEEGHIKNOQRRTUWY"
    solve_spaceword(puzzle_letters, "wCSW.txt")