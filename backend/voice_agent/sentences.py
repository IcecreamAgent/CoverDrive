"""Incremental sentence boundaries for low-latency, ordered speech synthesis."""

import re

# A dot at the end of a token may be the beginning of a decimal ("3." + "14").
# Wait for whitespace or flush() before treating an English period as final.
BOUNDARY = re.compile(r"[。！？!?；;\n]+|\.(?=\s)")


class SentenceBuffer:
    def __init__(self, max_chars: int = 160):
        self.pending = ""
        self.max_chars = max_chars

    def feed(self, delta: str) -> list[str]:
        self.pending += delta
        sentences = []
        while self.pending:
            match = BOUNDARY.search(self.pending)
            if match:
                end = match.end()
            elif len(self.pending) >= self.max_chars:
                # Prefer a clause boundary; cap unpunctuated output for bounded latency.
                end = max(self.pending.rfind(c, 0, self.max_chars) + 1 for c in "，,、 ")
                if end < self.max_chars // 2:
                    end = self.max_chars
            else:
                break
            sentence, self.pending = self.pending[:end].strip(), self.pending[end:]
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> str:
        sentence, self.pending = self.pending.strip(), ""
        return sentence
