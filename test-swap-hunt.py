import sys
import random
import time
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations, combinations
from multiprocessing import Manager
import traceback  # Import traceback module
from functools import lru_cache
from tdict import tdict

# ----------------- CONFIG (63-letter) -----------------
LETTER_COUNT = 63
GRID_ROWS = 7
GRID_COLS = 10
NUM_PROCESSES = 1 #GRID_ROWS
MAX_GENERATION_CYCLES = 20              # Maximum cycles to try generating anchors
CYCLE_REPORT_INTERVAL = 1
BACKTRACK_REPORT_INTERVAL = 100 #100
MAX_HASH_PLACEMENT_ATTEMPTS = 30            # Number of hash placement retries per anchor set
BACKTRACK_THRESHOLD_CHECKPOINTS = [2000, 3000, 5000, 10000, 20000]  # At these checkpoints, the coverage will be compared with the threshold.  Failing to surpass the threshold will trigger stop backtracking.
BACKTRACK_COVERAGE_THRESHOLD = [.70, .80, .90, .95, .96] # Minimum coverage threshold to continue backtracking
MAX_BACKTRACK_ATTEMPTS_PER_PATTERN = 20000   # Ultimate maximum backtrack attempts per pattern
SAVAGE_BACKTRACKING_THRESHOLD = .95 # Minimum coverage threshold to trigger savage backtracking (1 = no savage backtracking)

# ----------------- CONFIG (21-letter) -----------------
# LETTER_COUNT = 21
# GRID_ROWS = 3
# GRID_COLS = 8
# NUM_PROCESSES = 1
# MAX_GENERATION_CYCLES = 100 # Maximum cycles to try generating anchors
# CYCLE_REPORT_INTERVAL = 1
# BACKTRACK_REPORT_INTERVAL = 1 #100
# MAX_HASH_PLACEMENT_ATTEMPTS = 30         # Number of hash placement retries per anchor set
# BACKTRACK_THRESHOLD_CHECKPOINTS = [100, 200, 300]  # At these checkpoints, the coverage will be compared with the threshold.  Failing to surpass the threshold will trigger stop backtracking.
# BACKTRACK_COVERAGE_THRESHOLD = [.70, .80, .90] # Minimum coverage threshold to continue backtracking
# MAX_BACKTRACK_ATTEMPTS_PER_PATTERN = 1000   # Ultimate maximum backtrack attempts per pattern
# SAVAGE_BACKTRACKING_THRESHOLD = .9 # Minimum coverage threshold to trigger savage backtracking (1 = no savage backtracking)

# ---------- OTHER GLOBALS ----------
stop_backtracking = False

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

def is_valid_line(line, dict_by_length):
    """Check if a line (row or column) is valid."""
    for _, segment in get_segments(line):
        seg_str = ''.join(segment)
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

    def place(self, word, r, c, is_row):
        if is_row:
            if c + len(word) > self.cols:
                raise IndexError(f"Word '{word}' does not fit horizontally at row {r}, col {c}")
            for i, ch in enumerate(word):
                self.cells[r][c + i] = ch
        else:
            if r + len(word) > self.rows:
                raise IndexError(f"Word '{word}' does not fit vertically at row {r}, col {c}")
            for i, ch in enumerate(word):
                self.cells[r + i][c] = ch

    def insert_gap(self, r, pos):
        self.cells[r][pos] = '#'

    def coverage(self):
        filled = sum(ch != '.' for row in self.cells for ch in row)
        return filled / (self.rows * self.cols)

    def as_tuple(self):
        return tuple(''.join(row) for row in self.cells)

    def show(self):
        return "\n".join("".join(row) for row in self.cells)


# ---------- RANDOM HASH PLACEMENT on GRID ----------
def grid_hash_placement(grid, lex, max_hashes):
    """Generator: Yields each filled grid found for a random hash pattern."""

    candidate_grids = []

    for _ in range(10 * MAX_HASH_PLACEMENT_ATTEMPTS):
        g = grid.copy()
        empty = [(r, c) for r in range(g.rows) for c in range(g.cols) if g.cells[r][c] == '.']
        random.shuffle(empty)
        for pos in empty[:max_hashes]:
            g.cells[pos[0]][pos[1]] = '#'
        lengths = []
        for is_row in [True, False]:
            for idx in range(g.rows if is_row else g.cols):
                # Get the line (row or column)
                line = g.cells[idx] if is_row else [g.cells[r][idx] for r in range(g.rows)]
                for _, segment in get_segments(line):
                    pattern = ''.join(segment)
                    if lex.exists(pattern):
                        lengths.append(len(segment))
        if not lengths:
            continue
        score = sum(l ** 2 for l in lengths)
        candidate_grids.append((score, g))

    candidate_grids.sort(key=lambda x: x[0])

    for _, g in candidate_grids[:MAX_HASH_PLACEMENT_ATTEMPTS]:
        yield g  # Yield the filled grid

# ---------- ANCHOR GENERATION  ----------
def get_word_score(word, rare_dict):
    """
    Calculate the score of a word based on the rare_dict.
    The score is the sum of the weights (1 / frequency) of the rare letters in the word.
    """
    return sum(1 / rare_dict[ch] for ch in set(word) if ch in rare_dict)

def generate_anchor_sets(seed, puzzle_letter_counter, lex, rare_dict, seen, anchor_row_offset, manager_lock):
    print(f"\n[Seed {seed}] Starting anchor generation with seed {seed}")
    num_hashes = (GRID_ROWS * GRID_COLS) - LETTER_COUNT
    anchor_row = (seed + anchor_row_offset) % GRID_ROWS
    print(f"[Seed {seed}] Generating anchor sets for row {anchor_row} with {num_hashes} hashes.")
    # Build anchor row candidates once
    row_candidates = [w for w in lex.words[GRID_COLS]]
    print(f"[Seed {seed}] Anchor row candidates: {len(row_candidates)}")
    row_candidates.sort(
        key=lambda w: get_word_score(w, rare_dict),
        reverse=True
    )

    k = 10  # Number of top candidates to shuffle
    top_candidates = row_candidates[:k]
    print(f"[Seed {seed}] Shuffling top {k} anchor row candidates: {top_candidates}")
    random.shuffle(top_candidates)
    row_candidates = top_candidates + row_candidates[k:]

    total_attempts = [0]
    max_coverage_so_far = [0.00]
    swap_letters_tried = set()  # Track tried letter sets

    for anchor_row_idx, anchor_row_word in enumerate(row_candidates):
        cycle = anchor_row_idx + 1
        if cycle > MAX_GENERATION_CYCLES:
            break  # Stop after max cycles

        # Place anchor row word ONCE per anchor_row_word
        base_grid = Grid(GRID_ROWS, GRID_COLS)
        base_available = puzzle_letter_counter.copy()
        base_used_words = set()
        base_anchors = []

        base_grid.place(anchor_row_word, anchor_row, 0, is_row=True)
        use_word_on_segment(anchor_row_word, ['.'] * GRID_COLS, base_available)
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
                w for w in lex.words[GRID_ROWS]
                if w[anchor_row] == intersect_letter and can_use_word_on_segment(w, col_segment, available) and w not in used_words
            ]
            if not col_candidates:
                continue # next anchor_col
            col_candidates.sort(
                key=lambda w: get_word_score(w, rare_dict),
                reverse=True
            )
            anchor_col_word = col_candidates[0]

            # Place anchor column word
            grid.place(anchor_col_word, 0, anchor_col, is_row=False)
            use_word_on_segment(anchor_col_word, col_segment, available)
            used_words.add(anchor_col_word)
            anchors.append((anchor_col_word, 'col', anchor_col))

            # Check if enough empty cells for hashes
            empty_cells = sum(row.count('.') for row in grid.cells)
            remaining_hashes = num_hashes
            if empty_cells < remaining_hashes:
                raise ValueError(f"Not enough empty cells for hashes: {empty_cells} < {remaining_hashes}")

            for hash_attempts, grid_with_hashes in enumerate(grid_hash_placement(grid, lex, remaining_hashes), start=1):
                available_copy = available.copy()
                grid_tuple = grid_with_hashes.as_tuple()
                if manager_lock:
                    with manager_lock:
                        if grid_tuple in seen:
                            continue # next hash_attempts
                        seen.append(grid_tuple)
                else:
                    if grid_tuple in seen:
                        continue # next hash_attempts
                    seen.add(grid_tuple)

                total_attempts[0] = 0
                max_coverage_so_far[0] = 0.00
                savage_copy = []

                global stop_backtracking
                stop_backtracking = False

                if backtrack_fill(seed, grid_with_hashes, lex, rare_dict, available_copy, 0, total_attempts, cycle, anchor_col, hash_attempts, max_coverage_so_far, savage_copy):
                    print(f"[Seed {seed}] ✅ Solution Found:")
                    print(grid_with_hashes.show())
                    print(f"Anchors: {anchors}")
                    return [(grid_with_hashes, anchors)]

                if len(savage_copy) > 0:
                    for copy in savage_copy:
                        grid, available_for_swap = copy
                        letters_remained = ''.join([ch * count for ch, count in available_for_swap.items()])
                        if letters_remained in swap_letters_tried:
                            print(f"Skipping duplicate swap hunt for letters: {letters_remained}")
                            continue # next savage_copy
                        swap_letters_tried.add(letters_remained)  # Add to tried set
                        if swap_hunt(grid, letters_remained, lex.words):
                            print(f"[Seed {seed}] ✅ Solution Found:")
                            print(grid.show())
                            print(f"Anchors: {anchors}")
                            return [(grid, anchors)]
                        continue    # next savage_copy
                print(f"[Seed {seed}] No solution found for anchor row '{anchor_row_word}' at row {anchor_row}, col {anchor_col} after {hash_attempts} hash attempts.")
                continue  # next hash_attempts
            continue # next anchor_col
        continue  # next anchor_row_word (cycle)
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


def can_use_word_on_segment(word, segment, available):
    """Only require available letters for positions where segment has '.'."""
    word_count = Counter()
    for i, ch in enumerate(word):
        if segment[i] == '.':
            word_count[ch] += 1
    for ch, cnt in word_count.items():
        if available[ch] < cnt:
            return False
    return True

def use_word_on_segment(word, segment, available):
    """Decrement only the letters that are actually placed (i.e., where segment has '.')."""
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
def find_mrv_segment(grid, available, lex, rare_dict):
    """
    Optimized MRV heuristic: Prioritize segments with rare letters first, then longest segments.
    Returns a tuple: (is_row, row/col, start, candidates, pattern)
    """
    best = None
    rare_letters = set(rare_dict.keys()) if rare_dict else set()

    for is_row, outer_range, accessor in [
        (True, range(grid.rows), lambda idx: grid.cells[idx]),
        (False, range(grid.cols), lambda idx: [grid.cells[r][idx] for r in range(grid.rows)])
    ]:
        for idx in outer_range:
            line = accessor(idx)
            if '.' not in line:  # Skip fully filled rows/columns
                continue

            for start, segment in get_segments(line):
                seg_str = ''.join(segment)
                if len(seg_str) > 1 and '.' in seg_str and '#' not in seg_str:
                    # Check for rare letters
                    if rare_letters.intersection(seg_str):
                        candidates = [
                            w for w in lex.words[len(seg_str)]
                            if lex.word_matches_pattern(w, seg_str) and can_use_word_on_segment(w, seg_str, available)
                        ]
                        if candidates:
                            return (is_row, idx, start, candidates, seg_str)  # Immediately return rare letter segment

                    # Otherwise, consider the segment based on length
                    if not best or len(seg_str) > len(best[-1]):  # Compare segment lengths
                        candidates = [
                            w for w in lex.words[len(seg_str)]
                            if lex.word_matches_pattern(w, seg_str) and can_use_word_on_segment(w, seg_str, available)
                        ]
                        if candidates:
                            best = (is_row, idx, start, candidates, seg_str)

    return best  # Return the longest segment if no rare letter segment is found

def is_crossing_valid(line, lex):
    """Check if all segments crossing the given line are valid."""
    # print(f"Checking line: {line}")
    for start, segment in get_segments(line):
        seg_str = ''.join(segment)
        # print(f"Validating segment: {seg_str}, Start: {start}")
        if len(seg_str) > 1:
            if not lex.exists(seg_str):
                # print(f"Segment '{seg_str}' not found in dictionary.")
                return False
    return True

def swap_hunt(grid, initial_letters_str, dictionary_words):
    """
    Perform a swap hunt on the grid using the available letters (1 to n, n > 4 not advised).
    """
    # Extract available letters from initial_letters_str
    l_tuple = tuple(initial_letters_str)  # Initial l-tuple
    n = len(l_tuple)
    if n == 0:
        raise ValueError("swap_hunt requires at least 1 available letter.")

    # Get all empty and non-empty cells in the grid
    empty_cells = [(r, c) for r in range(grid.rows) for c in range(grid.cols) if grid.cells[r][c] == '.']
    non_empty_cells = [(r, c) for r in range(grid.rows) for c in range(grid.cols) if grid.cells[r][c] != '.']

    if len(empty_cells) < n or len(non_empty_cells) < n:
        raise ValueError("Not enough empty or non-empty cells to perform the swap.")

    # Initialize the first p-tuple (positions of empty cells)
    p_tuple = tuple(empty_cells[:n])  # First p-tuple
    tried_p_tuples = set()  # Set of tried p-tuples

    print(f"*********** Attempting swap hunt with {n} letters ************")
    print(f"Letters remained: {initial_letters_str}")
    print(f"Grid state before swap hunt:\n{grid.show()}")

    # Iterate over all combinations of n non-empty cells (ep-tuple)
    for ep_tuple in combinations(non_empty_cells, n):
        # Skip if this ep-tuple has already been tried
        if ep_tuple in tried_p_tuples:
            continue
        tried_p_tuples.add(ep_tuple)

        # # Extract the corresponding el-tuple (letters in ep-tuple positions)
        # el_tuple = tuple(grid.cells[r][c] for r, c in ep_tuple)
        first_permutation_printed = False

        # Iterate over all permutations of the l-tuple
        for permuted_l_tuple in permutations(l_tuple):
            print(f"Trying swap with l-tuple {permuted_l_tuple} in p-tuple {p_tuple} and ep-tuple {ep_tuple}")
            print(f"Grid state before swap:\n{grid.show()}")

            # Create a working copy of the grid
            working_grid = grid.copy()

            # Perform the swap
            # Place the letters from permuted_l_tuple into the positions in p-tuple
            for (r, c), letter in zip(p_tuple, permuted_l_tuple):
                working_grid.cells[r][c] = letter

            # Move the letters from ep-tuple into the positions in p-tuple
            for (r1, c1), (r2, c2) in zip(ep_tuple, p_tuple):
                working_grid.cells[r2][c2] = grid.cells[r1][c1]
                working_grid.cells[r1][c1] = '.'  # Clear the original non-empty cell
            if not first_permutation_printed:
                print(f"Grid state after swap:\n{working_grid.show()}")
                first_permutation_printed = True

            # Validate intersecting rows and columns
            rows_to_check = {r for r, _ in p_tuple + ep_tuple}
            cols_to_check = {c for _, c in p_tuple + ep_tuple}
            valid = True

            for r in rows_to_check:
                if not is_valid_line(working_grid.cells[r], dictionary_words):
                    valid = False
                    break
            if valid:
                for c in cols_to_check:
                    col = [working_grid.cells[row][c] for row in range(working_grid.rows)]
                    if not is_valid_line(col, dictionary_words):
                        valid = False
                        break

            # If all lines are valid, check the endgame condition
            if valid:
                print(f"Grid found valid after swap with l-tuple {permuted_l_tuple} and p-tuple {p_tuple}")
                print(working_grid.show())
                if check_endgame(working_grid, dictionary_words):
                    print("Valid swap found!")
                    print(working_grid.show())
                    return working_grid
            # else:
            #     l_tuple = el_tuple
  
    print(f"No valid swap found for grid:\n{grid.show()} and letters remained: {initial_letters_str}")
    return None

# ---------- Backtrack and Fill ----------
def backtrack_fill(seed, grid, lex, rare_dict, available, local_attempts, total_attempts, cycle, current_col, hash_attempt, max_coverage_so_far, savage_copy):

    global stop_backtracking
    if stop_backtracking:
        return False  # Immediately stop if the global flag is set

    if not any(available.values()):
        return check_endgame(grid, lex.words)

    empty = find_next_empty(grid)
    if not empty:
        return check_endgame(grid, lex.words)

    local_attempts += 1
    total_attempts[0] += 1  # Increment the first element of the list

    # Check if total attempts exceed the maximum allowed
    if total_attempts[0] > MAX_BACKTRACK_ATTEMPTS_PER_PATTERN:
        stop_backtracking = True  # Set the global flag
        return False

    current_coverage = grid.coverage()
    if local_attempts % BACKTRACK_REPORT_INTERVAL == 0:
        print(f"[Seed {seed}] Cycle {cycle}, Col {current_col}, Hash {hash_attempt}: Backtracking attempts: {local_attempts} (Total: {total_attempts[0]}, coverage: {current_coverage:.2f} | {max_coverage_so_far[0]:.2f})")
        print("Current grid state:")
        print(grid.show())

    # Check coverage at the threshold checkpoint
    max_coverage_so_far[0] = max(max_coverage_so_far[0], current_coverage)
    for checkpoint, coverage_threshold in zip(BACKTRACK_THRESHOLD_CHECKPOINTS, BACKTRACK_COVERAGE_THRESHOLD):
        if total_attempts[0] >= checkpoint:
            if max_coverage_so_far[0] < coverage_threshold:
                print(f"[Seed {seed}] Cycle {cycle}, Col {current_col}, Hash {hash_attempt}: Backtracking attempts exceeded checkpoint {checkpoint} with coverage {max_coverage_so_far[0]:.2f} (threshold: {coverage_threshold:.2f})")
                stop_backtracking = True  # Set the global flag
                return False
    
    # Save a savage copy for future swap hunts
    if current_coverage >= SAVAGE_BACKTRACKING_THRESHOLD:
        savage_copy.append((grid.copy(), available.copy()))
        print(f"Savage copy updated at coverage {max_coverage_so_far[0]:.2f}")
        print("Savage copy grid state:")
        print(savage_copy[-1][0].show())
        print(f"Savage copy available letters: {savage_copy[-1][1]}")

    mrv = find_mrv_segment(grid, available, lex, rare_dict)
    if not mrv:
        return False  # No fillable segment found

    is_row, idx, start, candidates, pattern = mrv
    # print(f"MRV Segment: is_row={is_row}, idx={idx}, start={start}, candidates={candidates}, pattern={pattern}")

    for word in candidates:
        new_grid = grid.copy()
        new_available = available.copy()

        # Correctly interpret idx and start based on is_row
        if is_row:
            segment = [grid.cells[idx][start + i] for i in range(len(word))]
        else:
            segment = [grid.cells[start + i][idx] for i in range(len(word))]

        # print(f"Placing word '{word}' at row {start if is_row else idx}, col {idx if is_row else start}, is_row {is_row}")

        # Validate placement boundaries
        if is_row and start + len(word) > grid.cols:
            print(f"Invalid placement: Word '{word}' exceeds grid width at row {idx}, col {start}")
            continue
        if not is_row and start + len(word) > grid.rows:
            print(f"Invalid placement: Word '{word}' exceeds grid height at row {start}, col {idx}")
            continue

        # Place the word
        new_grid.place(word, idx if is_row else start, start if is_row else idx, is_row)

        # Check all crossing segments for each filled cell
        valid = True
        for i in range(len(word)):
            if segment[i] == '.':
                crossing_line = (
                    [new_grid.cells[row][start + i] for row in range(new_grid.rows)] if is_row
                    else new_grid.cells[start + i]
                )
                # print(f"Crossing line: {crossing_line}, is_row: {is_row}")

                if not is_crossing_valid(crossing_line, lex):
                    valid = False
                    break
        if not valid:
            continue

        use_word_on_segment(word, segment, new_available)
      
        if backtrack_fill(seed, new_grid, lex, rare_dict, new_available, local_attempts, total_attempts, cycle, current_col, hash_attempt, max_coverage_so_far, savage_copy):
            grid.cells = new_grid.cells
            return True

    return False

# ---------- SOLVER ENTRY POINT WITH SUPPORT FOR MULTI-PROCESSING ----------
def run_solver(args):
    try:
        print(f"Running solver:")
        return generate_anchor_sets(*args)
    except Exception as e:
        # Print the full traceback to retain the original error position
        print("Fatal error in worker process:")
        print(traceback.format_exc())
        raise  # Re-raise the exception to propagate it

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
        # Build dictionary with length filter (3..6 letters)
    lex = tdict(dictionary_path, letters, len_min=1, len_max=GRID_COLS)
    rare_dict = lex.get_n_rarest_letters(n=3)

    for length in lex.words:
        print(f"Dictionary length {length}: {len(lex.words[length])} words")
    print(f"Available letters: {puzzle_letter_counter}")
    print(f"Letter frequency: {lex.letter_freq}")
    print(f"Rare letters: {rare_dict}")

    # Remove multiprocessing and directly call run_solver
    seen = set()  # Use a regular set for testing
    start_time = time.time()
    seeds = list(range(NUM_PROCESSES))
    anchor_row_offset = random.randint(1, GRID_ROWS)
    print(f"Anchor row offset: {anchor_row_offset}")
    seed_args = [
        (seed, puzzle_letter_counter, lex, rare_dict, seen, anchor_row_offset, None)
        for seed in seeds
    ]

    found = False
    print(f"\nStarting anchor generation and backtracking with {NUM_PROCESSES} processes (single-threaded)...")
    for args in seed_args:
        try:
            res = run_solver(args)
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
        except Exception as e:
            print("Fatal error in solver:", e)
            print(traceback.format_exc())
            sys.exit(1)

    if not found:
        print("❌ No solution found yet. Consider adjusting the parameters.")

# if __name__ == "__main__":
#     puzzle_letters = "AAAAABBBCCDDDEEEEEEEEFFFGGGHHHIILMMMNNOOOOPPPQRSSSSTTTTUUUVXYYZ"
#     # puzzle_letters = "ADEEEEEEGHIKNOQRRTUWY"
#     # puzzle_letters = "AACDEEEGHIIKNOQRRTUWY"
#     solve_spaceword(puzzle_letters, "wCSW.txt")


if __name__ == "__main__":
    # Provided variables
    puzzle_letters = "ABDEEEGIJLLMOOOOQSTUV"
    letters_remained = "JOO"
    grid_str = """
    Q#GOO.
    UT#M.L
    I#L.BE
    DEAVES
    """

    # Parse the grid from the string
    grid_lines = [line.strip() for line in grid_str.strip().split("\n")]
    rows = len(grid_lines)
    cols = len(grid_lines[0])
    grid = Grid(rows, cols)
    for r, line in enumerate(grid_lines):
        grid.cells[r] = list(line)

    # Initialize the dictionary
    dictionary_path = "wCSW.txt"  # Path to the dictionary file
    lex = tdict(dictionary_path, puzzle_letters, len_min=1, len_max=cols)

    # Run swap_hunt
    result = swap_hunt(grid, letters_remained, lex.words)

    # Print the result
    if result:
        print("Valid swap found!")
        print(result.show())
    else:
        print("No valid swap found.")