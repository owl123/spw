from collections import defaultdict, Counter
from typing import List, Optional, Dict, Iterable

ALPHABET_SIZE = 26
ALL_LETTERS_MASK = (1 << ALPHABET_SIZE) - 1

def lane_of(ch: str) -> int:
    o = ord(ch)
    if 65 <= o <= 90:  # 'A'..'Z'
        return o - 65
    raise ValueError(f"Unsupported char: {ch!r} (use A-Z)")

def to_lanes(word: str) -> Iterable[int]:
    for ch in word:
        yield lane_of(ch)

class BitmaskTrie:
    """
    32-way trie node (we only use lanes 0..25).
    - children: fixed array of size 32 for cache-friendly lookups
    - child_mask: 32-bit bitset, bit i set => child[i] exists
    - is_end: word terminator flag
    - subtree_count: number of words in this subtree (used to prune and count)
    """
    __slots__ = ("children", "child_mask", "is_end", "subtree_count")

    def __init__(self):
        self.children = [None] * 32
        self.child_mask = 0
        self.is_end = False
        self.subtree_count = 0

    def insert_word(self, word: str) -> None:
        node = self
        node.subtree_count += 1
        for lane in to_lanes(word):
            child = node.children[lane]
            if child is None:
                child = node.children[lane] = BitmaskTrie()
                node.child_mask |= (1 << lane)
            node = child
            node.subtree_count += 1
        node.is_end = True

    # ---------- Pattern handling ('.' wildcard only) ----------
    @staticmethod
    def compile_pattern(pattern: str) -> List[int]:
        """
        Returns a list of per-position masks.
        '.' => ALL_LETTERS_MASK; 'A'..'Z' => single-bit mask.
        """
        masks = []
        for c in pattern:
            if c == '.':
                masks.append(ALL_LETTERS_MASK)
            else:
                masks.append(1 << lane_of(c))
        return masks

    def find_all(self, pattern: str, limit: Optional[int] = None) -> List[str]:
        """Return up to `limit` words matching `pattern` ('.' wildcard)."""
        allowed = self.compile_pattern(pattern)
        n = len(allowed)
        out: List[str] = []
        # stack entries: (node, depth, built_chars_list)
        stack = [(self, 0, [])]

        while stack:
            node, i, acc = stack.pop()
            if node is None or node.subtree_count == 0:
                continue
            if i == n:
                if node.is_end:
                    out.append("".join(acc))
                    if limit is not None and len(out) >= limit:
                        break
                continue

            # Only branch into letters that both exist and are allowed here
            mask = (node.child_mask & allowed[i]) & ((1 << ALPHABET_SIZE) - 1)

            # Fast path: if remaining pattern positions are all '.' and there is a
            # single branch at this depth, we still need to walk to gather strings,
            # so keep normal iteration for correctness but keep pruning aggressively.
            while mask:
                lsb = mask & -mask
                lane = lsb.bit_length() - 1
                mask ^= lsb
                child = node.children[lane]
                if child is not None and child.subtree_count > 0:
                    stack.append((child, i + 1, acc + [chr(lane + 65)]))
        return out

    def count_matches(self, pattern: str) -> int:
        """Count words matching `pattern` without constructing them."""
        allowed = self.compile_pattern(pattern)
        n = len(allowed)

        # stack entries: (node, depth)
        stack = [(self, 0)]
        total = 0

        # Optimization: quick test if the tail is all '.'.
        # If at some depth d we see all remaining masks == ALL_LETTERS_MASK,
        # we can add child.subtree_count directly for each branch and stop exploring deeper.
        tail_all_any = [False] * (n + 1)
        tail_all_any[n] = True
        acc = True
        for i in range(n - 1, -1, -1):
            acc = acc and (allowed[i] == ALL_LETTERS_MASK)
            tail_all_any[i] = acc

        while stack:
            node, i = stack.pop()
            if node is None or node.subtree_count == 0:
                continue
            if i == n:
                if node.is_end:
                    total += 1
                continue

            mask_here = (node.child_mask & allowed[i]) & ((1 << ALPHABET_SIZE) - 1)
            if mask_here == 0:
                continue

            # If the rest are all '.' we can sum subtree_count without deeper traversal
            if tail_all_any[i + 1]:
                # Sum subtree_count of all valid child branches directly
                while mask_here:
                    lsb = mask_here & -mask_here
                    lane = lsb.bit_length() - 1
                    mask_here ^= lsb
                    child = node.children[lane]
                    if child is not None and child.subtree_count > 0:
                        total += child.subtree_count
                continue

            # Otherwise, keep traversing
            while mask_here:
                lsb = mask_here & -mask_here
                lane = lsb.bit_length() - 1
                mask_here ^= lsb
                child = node.children[lane]
                if child is not None and child.subtree_count > 0:
                    stack.append((child, i + 1))
        return total
    
    def contains(self, word: str) -> bool:
        """Exact word lookup (no wildcards)."""
        node = self
        for lane in to_lanes(word):
            node = node.children[lane]
            if node is None:
                return False
        return node.is_end

    def matches(self, pattern: str) -> bool:
        """
        Pattern existence with '.' wildcard.
        Early-exit as soon as a match is found (faster than count>0).
        """
        allowed = self.compile_pattern(pattern)
        n = len(allowed)
        stack = [(self, 0)]

        while stack:
            node, i = stack.pop()
            if node is None or node.subtree_count == 0:
                continue
            if i == n:
                if node.is_end:
                    return True
                continue

            mask = (node.child_mask & allowed[i]) & ((1 << ALPHABET_SIZE) - 1)
            while mask:
                lsb = mask & -mask
                lane = lsb.bit_length() - 1
                mask ^= lsb
                child = node.children[lane]
                if child is not None and child.subtree_count > 0:
                    stack.append((child, i + 1))
        return False 


class tdict:
    """
    Trie-backed dictionary partitioned by word length.
    - words[length] -> list of words (for optional external use)
    - tries[length] -> BitmaskTrie for that length
    - word_to_bitmask (optional; not required for current operations)
    """
    def __init__(self, path: str, available_letters: str = '', len_max: int = 99, len_min: int = 1):
        self.words: Dict[int, List[str]] = defaultdict(list)
        self.word_to_bitmask: Dict[str, int] = {}
        self.tries: Dict[int, BitmaskTrie] = {}
        self.load_from_file(path, available_letters, len_max, len_min)

    # ------------- Public API -------------
    def match(self, pattern: str, limit: Optional[int] = None) -> List[str]:
        """Return words matching pattern ('.' wildcard)."""
        L = len(pattern)
        trie = self.tries.get(L)
        if not trie:
            return []
        return trie.find_all(pattern, limit=limit)

    def count(self, pattern: str) -> int:
        """Count words matching pattern ('.' wildcard)."""
        L = len(pattern)
        trie = self.tries.get(L)
        if not trie:
            return 0
        return trie.count_matches(pattern)

    # ------------- Loading & helpers -------------
    @staticmethod
    def _clean_word(raw: str) -> Optional[str]:
        w = raw.strip().upper()
        if not w:
            return None
        # Only accept pure A-Z words
        for c in w:
            if not (65 <= ord(c) <= 90):
                return None
        return w

    @staticmethod
    def _can_build(word: str, avail: Counter) -> bool:
        if not avail:
            return True
        need = Counter(word)
        # early reject if any letter exceeds available multiplicity
        for ch, cnt in need.items():
            if cnt > avail.get(ch, 0):
                return False
        return True

    def _insert(self, word: str) -> None:
        L = len(word)
        self.words[L].append(word)
        trie = self.tries.get(L)
        if trie is None:
            trie = self.tries[L] = BitmaskTrie()
        trie.insert_word(word)

        # Optional: pack 5-bit lanes for the word (not used by match/count now)
        packed = 0
        shift = 0
        for lane in to_lanes(word):
            packed |= (lane & 31) << shift
            shift += 5
        self.word_to_bitmask[word] = packed  # (kept for potential future use)

    def load_from_file(self, path: str, available_letters: str = '', len_max: int = 99, len_min: int = 1) -> None:
        avail = Counter(available_letters) if available_letters else Counter()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = self._clean_word(line)
                if not w:
                    continue
                L = len(w)
                if L < len_min or L > len_max:
                    continue
                if not self._can_build(w, avail):
                    continue
                self._insert(w)

    def exists_word(self, word: str) -> bool:
        """
        Exact membership test (fastest path).
        """
        L = len(word)
        trie = self.tries.get(L)
        if not trie:
            return False
        return trie.contains(word)

    def exists_pattern(self, pattern: str) -> bool:
        """
        Pattern existence with '.' wildcard; early-exits on first match.
        """
        L = len(pattern)
        trie = self.tries.get(L)
        if not trie:
            return False
        return trie.matches(pattern)

    # Optional convenience: dispatch based on presence of '.'
    def exists(self, s: str) -> bool:
        """
        Convenience wrapper:
        - if s contains '.', treat as pattern
        - otherwise, exact-word membership
        """
        if '.' in s:
            return self.exists_pattern(s)
        return self.exists_word(s)            
    
    @staticmethod
    def word_matches_pattern(word: str, pattern: str) -> bool:
        """
        Return True if word matches pattern with '.' as wildcard.
        Example: 'APPLE' matches 'AP..E' but 'APRON' does not.
        """
        if len(word) != len(pattern):
            return False
        for wc, pc in zip(word, pattern):
            if pc != '.' and wc != pc:
                return False
        return True
    # ---------------- for backward compatibility
    def compute_letter_frequencies(self) -> Counter:
        """
        Compute frequency of each letter across the loaded words.
        - A letter counts once per *word* (set(word)), not per occurrence.
          e.g. 'APPLE' contributes A,P,L,E, but not multiple P's.
        """
        freq = Counter()
        for words in self.words.values():
            for word in words:
                freq.update(set(word))   # use set(word) to count distinct letters per word
        return freq

    def get_n_rarest_letters(self, n: int = 4) -> dict[str, int]:
        """
        Return a dict of the n rarest letters with their frequencies.
        """
        if not hasattr(self, "letter_freq"):
            self.letter_freq = self.compute_letter_frequencies()
        rare_items = sorted(self.letter_freq.items(), key=lambda x: x[1])[:n]
        rare_dict = {ch: freq for ch, freq in rare_items}
        return rare_dict