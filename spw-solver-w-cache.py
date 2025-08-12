import os
import random
import time
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from multiprocessing import Manager

# ----------------- CONFIG (63-letter) -----------------
LETTER_COUNT = 63
GRID_ROWS = 8
GRID_COLS = 10
NUM_PROCESSES = 1 #GRID_ROWS
MAX_GENERATION_CYCLES = 10000 # Maximum cycles to try generating anchors
CYCLE_REPORT_INTERVAL = 1
BACKTRACK_REPORT_INTERVAL = 1 #100
MAX_HASH_PLACEMENT_ATTEMPTS = 500         # Number of hash placement retries per anchor set
MAX_BACKTRACK_ATTEMPTS_PER_PATTERN = 10**5  # Adjust based on problem size and resources
MAX_FAILED_STATE_CACHE = 10**5  # Maximum number of failed states to remember

# ----------------- CONFIG (21-letter) -----------------
# LETTER_COUNT = 21
# GRID_ROWS = 3
# GRID_COLS = 8
# NUM_PROCESSES = 1
# MAX_GENERATION_CYCLES = 1000 # Maximum cycles to try generating anchors
# CYCLE_REPORT_INTERVAL = 1
# BACKTRACK_REPORT_INTERVAL = 1 #100
# MAX_HASH_PLACEMENT_ATTEMPTS = 100         # Number of hash placement retries per anchor set
# MAX_BACKTRACK_ATTEMPTS_PER_PATTERN = 10**5  # Adjust based on problem size and resources
# MAX_FAILED_STATE_CACHE = 10**4  # Maximum number of failed states to remember

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
def compute_letter_frequencies(dict_by_length):
    freq = Counter()
    for words in dict_by_length.values():
        for word in words:
            freq.update(set(word))
    return freq

def get_n_rarest_letters(dict_letter_freq, n=4):
    """Return a list of the N rarest letters in the dictionary."""
    return [ch for ch, _ in sorted(dict_letter_freq.items(), key=lambda x: x[1])[:n]]

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
    dict_letter_freq = compute_letter_frequencies(dict_by_length)
    return dict_by_length, mask_dict, dict_letter_freq

# ---------- RANDOM HASH PLACEMENT on GRID ----------
def grid_hash_placement(grid, dict_by_length, max_hashes, mask_dict, dict_letter_freq):
    """
    Generator: Yields each filled grid found for a random hash pattern.
    """
    rare_letters = get_n_rarest_letters(dict_letter_freq, n=4)
    candidate_grids = []

    for _ in range(10 * MAX_HASH_PLACEMENT_ATTEMPTS):
        g = grid.copy()
        empty = [(r, c) for r in range(g.rows) for c in range(g.cols) if g.cells[r][c] == '.']
        random.shuffle(empty)
        for pos in empty[:max_hashes]:
            g.cells[pos[0]][pos[1]] = '#'
        lengths = []
        for r in range(g.rows):
            row = g.cells[r]
            for _, segment in get_segments(row):
                if '#' not in segment and len(segment) > 1:
                    if all(ch == '.' for ch in segment):
                        if len(dict_by_length[len(segment)]) > 0:
                            lengths.append(len(segment))
                    else:
                        pattern_mask = pattern_to_mask(segment)
                        if matches_any(pattern_mask, mask_dict, len(segment)):
                            lengths.append(len(segment))
        for c in range(g.cols):
            col = [g.cells[r][c] for r in range(g.rows)]
            for _, segment in get_segments(col):
                if '#' not in segment and len(segment) > 1:
                    if all(ch == '.' for ch in segment):
                        if len(dict_by_length[len(segment)]) > 0:
                            lengths.append(len(segment))
                    else:
                        pattern_mask = pattern_to_mask(segment)
                        if matches_any(pattern_mask, mask_dict, len(segment)):
                            lengths.append(len(segment))
        if not lengths:
            continue
        score = sum(l ** 2 for l in lengths)
        candidate_grids.append((score, g))

    candidate_grids.sort(key=lambda x: x[0])

    for _, g in candidate_grids[:MAX_HASH_PLACEMENT_ATTEMPTS]:
        yield g  # Yield the filled grid

# ---------- ANCHOR GENERATION (NEW LOGIC) ----------
def anchor_word_score(word, dict_letter_freq, n=2):
    freqs = sorted(dict_letter_freq[ch] for ch in set(word))
    return sum(freqs[:n])

def generate_anchor_sets(seed, puzzle_letter_counter, mask_dict, dict_by_length, dict_letter_freq, seen, anchor_row_offset):
    num_hashes = (GRID_ROWS * GRID_COLS) - LETTER_COUNT
    anchor_row = (seed + anchor_row_offset) % GRID_ROWS

    # Build anchor row candidates once
    row_candidates = [w for w in dict_by_length[GRID_COLS] if can_use_word(w, puzzle_letter_counter)]
    # TODO: add logic to handle len(row_candidates) = 0
    row_candidates.sort(key=lambda w: anchor_word_score(w, dict_letter_freq, n=2))
    # TODO: add logic to shuffle top 10 to add variability
    total_attempts = [0]

    for anchor_row_idx, anchor_row_word in enumerate(row_candidates):
        cycle = anchor_row_idx + 1
        if cycle >= MAX_GENERATION_CYCLES:
            break

        # Place anchor row word ONCE per anchor_row_word
        base_grid = Grid(GRID_ROWS, GRID_COLS)
        base_available = puzzle_letter_counter.copy()
        base_used_words = set()
        base_anchors = []

        base_grid.place(anchor_row_word, anchor_row, 0, 'H')
        use_word_on_segment(anchor_row_word, ['.']*GRID_COLS, base_available)
        base_used_words.add(anchor_row_word)
        base_anchors.append((anchor_row_word, 'row', anchor_row))

        for anchor_col in range(GRID_COLS):
        # Copy the grid and state after anchor row placement
            grid = base_grid.copy()
            available = base_available.copy()
            used_words = base_used_words.copy()
            anchors = base_anchors.copy()

            # Find anchor column word that intersects at anchor_col
            intersect_letter = anchor_row_word[anchor_col]
            col_segment = [grid.cells[r][anchor_col] for r in range(GRID_ROWS)]
            col_candidates = [
                w for w in dict_by_length[GRID_ROWS]
                if w[anchor_row] == intersect_letter and can_use_word_on_segment(w, col_segment, available) and w not in used_words
            ]
            if not col_candidates:
                continue
            col_candidates.sort(key=lambda w: anchor_word_score(w, dict_letter_freq, n=2))
            # TODO: add logic to shuffle top 10 to add variability
            anchor_col_word = col_candidates[0]

            # Place anchor column word
            grid.place(anchor_col_word, 0, anchor_col, 'V')
            use_word_on_segment(anchor_col_word, col_segment, available)  # Only mark new letters as used
            used_words.add(anchor_col_word)
            anchors.append((anchor_col_word, 'col', anchor_col))

            # Check if enough empty cells for hashes
            empty_cells = sum(row.count('.') for row in grid.cells)
            remaining_hashes = num_hashes
            if empty_cells < remaining_hashes:
                continue

            hash_attempts = 0
            for grid_with_hashes in grid_hash_placement(grid, dict_by_length, remaining_hashes, mask_dict, dict_letter_freq):
                grid_tuple = grid_with_hashes.as_tuple()
                if grid_tuple in seen:
                    continue
                seen.append(grid_tuple)
                total_attempts[0] += 1
                hash_attempts += 1

                if backtrack_fill(seed, grid_with_hashes, dict_by_length, mask_dict, available.copy(), 0, total_attempts, cycle, anchor_col, hash_attempts, rare_letters=get_n_rarest_letters(dict_letter_freq, n=4)):
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

def can_use_word_on_segment(word, segment, available):
    """
    Only require available letters for positions where segment has '.'.
    """
    word_count = Counter()
    for i, ch in enumerate(word):
        if segment[i] == '.':
            word_count[ch] += 1
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
def find_mrv_segment(grid, available, dict_by_length, mask_dict, rare_letters=None):
    """
    Find the segment (horizontal or vertical) with the fewest candidate words (MRV heuristic).
    If rare_letters is provided, immediately return the first segment containing a rare letter.
    Returns a tuple: (direction, row/col, start, candidates, pattern)
    """
    min_score = float('inf')
    best = None

    for direction, outer_range, accessor in [
        ('H', range(grid.rows), lambda idx: grid.cells[idx]),
        ('V', range(grid.cols), lambda idx: [grid.cells[r][idx] for r in range(grid.rows)])
    ]:
        for idx in outer_range:
            line = accessor(idx)
            for start, segment in get_segments(line):
                seg_str = ''.join(segment)
                if len(seg_str) > 1 and '.' in seg_str and '#' not in seg_str:
                    pattern_mask = pattern_to_mask(seg_str)
                    candidates = [
                        w for w in dict_by_length[len(seg_str)]
                        if mask_matches_pattern(word_to_mask(w), pattern_mask, len(seg_str)) and can_use_word_on_segment(w, seg_str, available)
                    ]
                    if len(candidates) == 0:
                        return None
                    # Rare letter priority: return immediately if segment contains a rare letter
                    if rare_letters and any(ch in seg_str for ch in rare_letters):
                        return (direction, idx, start, candidates, seg_str)
                    score = len(candidates) / (len(seg_str) ** 2)
                    if score < min_score:
                        min_score = score
                        best = (direction, idx, start, candidates, seg_str)

    return best  # or None if no fillable segment found

def is_crossing_valid(grid, dict_by_length, mask_dict, r, c, direction):
    # Check the crossing segment at (r, c)
    if direction == 'H':
        # Check vertical segment
        col = [grid.cells[row][c] for row in range(grid.rows)]
        for start, segment in get_segments(col):
            if start <= r < start + len(segment):
                seg_str = ''.join(segment)
                if len(seg_str) > 1:
                    if '.' not in seg_str:
                        if seg_str not in dict_by_length[len(seg_str)]:
                            return False
                    else:
                        pattern_mask = pattern_to_mask(seg_str)
                        if not matches_any(pattern_mask, mask_dict, len(seg_str)):
                            return False
                break
    else:
        # Check horizontal segment
        row = grid.cells[r]
        for start, segment in get_segments(row):
            if start <= c < start + len(segment):
                seg_str = ''.join(segment)
                if len(seg_str) > 1:
                    if '.' not in seg_str:
                        if seg_str not in dict_by_length[len(seg_str)]:
                            return False
                    else:
                        pattern_mask = pattern_to_mask(seg_str)
                        if not matches_any(pattern_mask, mask_dict, len(seg_str)):
                            return False
                break
    return True

def backtrack_fill(seed, grid, dict_by_length, mask_dict, available, local_attempt, total_attempt, cycle, current_col, hash_attempt, failed_states=None, rare_letters=None):
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

    local_attempt += 1
    total_attempt[0] += 1
    # Check if total attempts exceed the maximum allowed
    if total_attempt[0] > MAX_BACKTRACK_ATTEMPTS_PER_PATTERN:
        print(f"[Seed {seed}] Exceeded maximum backtracking attempts ({MAX_BACKTRACK_ATTEMPTS_PER_PATTERN}). "
            f"Cycle: {cycle}, Col: {current_col}, Hash Attempt: {hash_attempt}. Terminating backtracking.")
        return False

    if local_attempt % BACKTRACK_REPORT_INTERVAL == 0:
        print(f"[Seed {seed}] Cycle {cycle}, Col {current_col}, Hash {hash_attempt}: Backtracking attempts: {local_attempt} (Total: {total_attempt[0]}, coverage: {grid.coverage():.2f})")
        print("Current grid state:")
        print(grid.show())
        print("-" * 20)

    mrv = find_mrv_segment(grid, available, dict_by_length, mask_dict, rare_letters=rare_letters)
    if not mrv:
        failed_states[state_key] = None
        if len(failed_states) > MAX_FAILED_STATE_CACHE:
            failed_states.popitem(last=False)
        return False  # No fillable segment found

    direction, idx, start, candidates, pattern = mrv

    if local_attempt % BACKTRACK_REPORT_INTERVAL == 0:
        print(f"[Seed {seed}] MRV: {direction} {idx}, start {start}, pattern '{pattern}': "
              f"Backtracking attempts: {local_attempt} (Total: {total_attempt[0]}, coverage: {grid.coverage():.2f})")
        print("Current grid state:")
        print(grid.show())
        print("-" * 20)

    for word in candidates:
        new_grid = grid.copy()
        new_available = available.copy()
        if direction == 'H':
            segment = [grid.cells[idx][start + i] for i in range(len(word))]
            new_grid.place(word, idx, start, 'H')
            # Check all crossing vertical segments for each filled cell
            valid = True
            for i in range(len(word)):
                if segment[i] == '.':
                    if not is_crossing_valid(new_grid, dict_by_length, mask_dict, idx, start + i, 'H'):
                        valid = False
                        break
            if not valid:
                continue
        else:
            segment = [grid.cells[start + i][idx] for i in range(len(word))]
            new_grid.place(word, start, idx, 'V')
            # Check all crossing horizontal segments for each filled cell
            valid = True
            for i in range(len(word)):
                if segment[i] == '.':
                    if not is_crossing_valid(new_grid, dict_by_length, mask_dict, start + i, idx, 'V'):
                        valid = False
                        break
            if not valid:
                continue

        if is_valid_grid_segments(new_grid, dict_by_length):
            use_word_on_segment(word, segment, new_available)
            if backtrack_fill(seed, new_grid, dict_by_length, mask_dict, new_available, local_attempt, total_attempt, cycle, current_col, hash_attempt, failed_states, rare_letters=rare_letters):
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
    dict_by_length, mask_dict, dict_letter_freq = preprocess_dictionary(dictionary_path, puzzle_letter_counter)
    for length in dict_by_length:
        print(f"Dictionary length {length}: {len(dict_by_length[length])} words")

    print("\nStarting anchor generation and backtracking...")
    manager = Manager()
    seen = manager.list()  # Shared set to avoid duplicate grids
    start_time = time.time()
    seeds = list(range(NUM_PROCESSES))
    anchor_row_offset = random.randint(1, GRID_ROWS)
    seed_args = [
        (seed, puzzle_letter_counter, mask_dict, dict_by_length, dict_letter_freq, seen, anchor_row_offset)
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
    # puzzle_letters = "AACDEEEGHIIKNOQRRTUWY"
    solve_spaceword(puzzle_letters, "wCSW.txt")