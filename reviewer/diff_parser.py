"""Parse a unified diff into something the LLM can reason about AND we can validate against.

Two outputs per file:
  * `render()`         -> prompt text with explicit NEW-file line numbers
  * `commentable_lines`-> the set of line numbers a comment may legally point to

The second one is the hallucination filter: a comment on a line that isn't in the
diff is either wrong or unverifiable, so we drop it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from unidiff import PatchSet

DEFAULT_EXTENSIONS = (".py",)


@dataclass
class FileDiff:
    path: str
    added_lines: set[int] = field(default_factory=set)
    commentable_lines: set[int] = field(default_factory=set)  # added + context
    rendered_hunks: list[str] = field(default_factory=list)

    def render(self) -> str:
        return f"### FILE: {self.path}\n" + "\n".join(self.rendered_hunks)


def parse_diff(diff_text: str, extensions: tuple[str, ...] = DEFAULT_EXTENSIONS) -> list[FileDiff]:
    """Parse a unified diff. Keeps only modified/added files with matching extensions."""
    patch = PatchSet(diff_text)
    files: list[FileDiff] = []

    for pf in patch:
        if pf.is_removed_file or pf.is_binary_file:
            continue
        path = pf.path  # new path for renames
        if extensions and not path.endswith(extensions):
            continue

        fd = FileDiff(path=path)
        for hunk in pf:
            out = [f"@@ {hunk.section_header or ''}".rstrip()]
            for line in hunk:
                if line.is_removed:
                    # Removed lines have no new-file number; show them for context only.
                    out.append(f"     - {line.value.rstrip()}")
                    continue
                n = line.target_line_no
                fd.commentable_lines.add(n)
                if line.is_added:
                    fd.added_lines.add(n)
                    out.append(f"{n:>4} + {line.value.rstrip()}")
                else:
                    out.append(f"{n:>4}   {line.value.rstrip()}")
            fd.rendered_hunks.append("\n".join(out))

        if fd.added_lines:  # nothing new to review otherwise
            files.append(fd)

    return files


def render_for_prompt(files: list[FileDiff]) -> str:
    return "\n\n".join(f.render() for f in files)
