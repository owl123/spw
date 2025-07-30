#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, time
from collections import Counter, defaultdict, deque
from itertools   import combinations
from math        import comb
from pathlib     import Path
from typing      import List, Tuple

# ─── defaults ───────────────────────────────────────────────
DEF_DICT = "Projects/SpaceWord/wCSW.txt"          # change on Windows
#DEF_POOL = "AAACDDEEEGIIJMNOORRWZ"          # 21 tiles
#DEF_POOL = 'AAABCDEEEIKLTTTUUUVWY' #250703
DEF_POOL = 'ADEEEEEEGHIKNOQRRTUWY' #250704
#DEF_POOL = 'AAABBBCCCDEEEEEEEFFFGGGGHHIIIIIKMMMNNNOOOOPPQRRRSSSSTTTTUUUWXYY' #250629W
# ────────────────────────────────────────────────────────────

# ---------- dictionary loader ------------------------------ #
def load_words(path: str, pool: Counter, max_len: int):
    buckets: dict[int, List[str]] = defaultdict(list)
    need = pool
    with Path(path).open(encoding="utf-8", errors="ignore") as fh:
        for w in map(str.strip, fh):
            w = w.upper()
            if len(w) <= max_len and w.isalpha() and not (Counter(w) - need):
                buckets[len(w)].append(w)
    return {k: tuple(v) for k, v in buckets.items()}

# ---------- mask tables ------------------------------------ #
def build_masks(words_by_len: dict[int, Tuple[str, ...]]):
    masks, m2w = {}, {}
    for L, bucket in words_by_len.items():
        mset, d = set(), defaultdict(list)
        for w in bucket:
            for bits in range(1 << L):
                s = list(w)
                for i in range(L):
                    if bits & (1 << i):
                        s[i] = '.'
                m = ''.join(s)
                mset.add(m)
                d[m].append(w)
        mset.add('.' * L)
        d['.' * L] = bucket
        masks[L] = mset
        m2w[L] = {k: tuple(v) for k, v in d.items()}
    return masks, m2w

# ---------- island connectivity ---------------------------- #
def connected(blocks, R, C):
    blank = [True] * (R * C)
    for k in blocks:
        blank[k] = False
    try:
        start = blank.index(True)
    except ValueError:
        return False
    q, seen = deque([start]), {start}
    while q:
        k = q.popleft()
        r, c = divmod(k, C)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                idx = nr * C + nc
                if blank[idx] and idx not in seen:
                    seen.add(idx)
                    q.append(idx)
    return len(seen) == R * C - len(blocks)

# ---------- slot enumerator ------------------------------- #
def slots(grid, horiz):
    R, C = len(grid), len(grid[0])
    outer, inner = (R, C) if horiz else (C, R)
    cell = (lambda i, j: grid[i][j]) if horiz else (lambda i, j: grid[j][i])
    out = []
    for i in range(outer):
        run = 0
        for j in range(inner):
            if cell(i, j) is None:
                if run == 0:
                    start = j
                run += 1
            else:
                if run:
                    out.append(((i, start) if horiz else (start, i), run, horiz))
                    run = 0
        if run:
            out.append(((i, inner - run) if horiz else (inner - run, i), run, horiz))
    return out

def build_mask(grid, r, c, L, horiz):
    if horiz:
        return ''.join(grid[r][c + i] or '.' for i in range(L))
    return ''.join(grid[r + i][c] or '.' for i in range(L))

def down_ok(grid, slots_all, masks, m2w, pool):
    for (r, c), L, h in slots_all:
        if h or L == 1:
            continue
        mask = build_mask(grid, r, c, L, False)
        if '.' not in mask:
            if mask not in masks[L]:
                return False
        else:
            bucket = m2w[L].get(mask, ())
            dots = [i for i, ch in enumerate(mask) if ch == '.']
            need_ok = lambda w: not (Counter(w[i] for i in dots) - pool)
            if not any(need_ok(w) for w in bucket):
                return False
    return True

# ---------- solve a single pattern ------------------------ #
def solve_pattern(blocks, R, C, wb, masks, m2w,
                  pool0: Counter, cache_on: bool):
    grid = [[None] * C for _ in range(R)]
    for k in blocks:
        r, c = divmod(k, C)
        grid[r][c] = '#'

    rows = slots(grid, True)
    rows.sort(key=lambda s: -s[1])          # longest first
    cols = slots(grid, False)               # only for vertical pruning

    dom_cache: dict[tuple, list[str]] = {}
    mask_cache: dict[Tuple[int, str], Tuple[str, ...]] = {}
    hits = stores = 0

    def domain(slot, pool: Counter) -> list[str]:
        nonlocal hits, stores
        (r, c), L, _ = slot
        if L == 1:
            # Accept any single letter from the pool for singleton slots
            return [ch for ch in pool.elements() if pool[ch] > 0]

        mask = build_mask(grid, r, c, L, True)
        key = (slot, tuple(sorted(pool.items())), mask)

        if cache_on and key in dom_cache:
            hits += 1
            return dom_cache[key]

        bucket = mask_cache.get((L, mask))
        if bucket is None:
            bucket = m2w[L].get(mask, ())
            mask_cache[(L, mask)] = bucket

        if '.' not in mask:
            dom = [mask] if bucket else []
        else:
            dots = [i for i, ch in enumerate(mask) if ch == '.']
            need_ok = lambda w: not (Counter(w[i] for i in dots) - pool)
            dom = [w for w in bucket if need_ok(w)]

        # cache only if slot length ≥ 3 and >1 options
        if cache_on and L >= 3 and len(dom) > 1:
            dom_cache[key] = dom
            stores += 1
        return dom

    def backtrack(idx, pool):
        if idx == len(rows):
            return True
        slot = rows[idx]
        for w in domain(slot, pool):
            (r, c), L, _ = slot
            for i, ch in enumerate(w):
                grid[r][c + i] = ch
            new_pool = pool - Counter(w)
            if down_ok(grid, rows + cols, masks, m2w, new_pool):
                if backtrack(idx + 1, new_pool):
                    return True
            for i in range(L):
                grid[r][c + i] = None
        return False

    solved = backtrack(0, pool0.copy())
    return ([''.join(ch or '#' for ch in row) for row in grid] if solved else None,
            hits, stores)

# ---------- CLI driver ------------------------------------ #
def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--rows',type=int,default=4)
    pa.add_argument('--cols',type=int,default=6)
    pa.add_argument('--pool',type=str,default=DEF_POOL)
    pa.add_argument('--word-file',type=str,default=DEF_DICT)
    pa.add_argument('--show',type=int,default=1)
    pa.add_argument('--no-cache',action='store_true',
                    help='disable domain cache (ON by default)')
    pa.add_argument('--test-pattern',type=str,default=None,
                    help='comma-separated list of block indices to test a specific pattern (e.g. 0,9,19)')
    args=pa.parse_args()

    R,C=args.rows,args.cols
    if R*C<21: sys.exit("Board too small.")
    blocks=R*C-21
    pool=Counter(args.pool.upper())
    wb=load_words(args.word_file,pool,max(R,C))
    masks,m2w=build_masks(wb)

    use_cache = not args.no_cache

    if args.test_pattern:
        pat = tuple(int(x) for x in args.test_pattern.split(','))
        if len(pat) != blocks:
            print(f"Error: pattern length {len(pat)} does not match number of blocks {blocks}.")
            sys.exit(1)
        print(f"Testing specific pattern: {pat}")
        start = time.perf_counter()
        grid,hits,stores=solve_pattern(pat,R,C,wb,masks,m2w,pool,use_cache)
        dt=(time.perf_counter()-start)*1000
        print(f"pattern {pat}{'*' if grid else ''}: {dt:7.1f} ms   cache hits {hits:6}, stores {stores:6}")
        if grid:
            print(f"\n*** SOLVED  blocks {pat} ***\n"+'\n'.join(grid))
            print(f"Solution found for test pattern.")
        else:
            print(f"No solution found for test pattern.")
        return

    legal=[p for p in combinations(range(R*C),blocks) if connected(p,R,C)]
    print(f"{R}×{C} board | {blocks} blocks | "
          f"{len(legal):,}/{comb(R*C,blocks):,} patterns")
    print("domain-cache:", "ON" if use_cache else "OFF",
          "| slot order: ACROSS rows (longest first)")

    t0=time.perf_counter()
    found=0
    for n,pat in enumerate(legal,1):
        start=time.perf_counter()
        grid,hits,stores=solve_pattern(pat,R,C,wb,masks,m2w,pool,use_cache)
        dt=(time.perf_counter()-start)*1000
        print(f"pattern {n:3} {pat}{'*' if grid else ''}: {dt:7.1f} ms   cache hits {hits:6}, stores {stores:6}")
        if grid:
            found += 1
            print(f"\n*** SOLVED  blocks {pat} ***\n"+'\n'.join(grid))
            if args.show:
                print(f"Solution: {found} of {args.show} requested (use --show=n to change)\n")
            if args.show and found>=args.show: break
    print(f"\n{n} patterns tested in {time.perf_counter()-t0:.2f}s")

if __name__ == "__main__":
    main()