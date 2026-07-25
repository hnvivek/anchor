"""cite - demo the citation layer: turn an AI answer into a cited, flagged one.

Each grounded claim gets an inline [n] pointing to the exact source doc + char
span; ungrounded claims get a warning. Spell-check, but for facts - with
footnotes you can click through and verify.

    run:  python3 cite.py
"""

from anchor import Source, annotate, render_markdown

policy = Source(
    id="PTO-Policy",
    text=("Employees get 15 days of PTO per year. "
          "Up to 5 unused days roll over into the next year. "
          "Requests must be submitted 7 days in advance."),
)

# an AI answer: two true sentences + one fabricated
answer = (
    "You can roll over up to 5 unused days into next year. "
    "To request time off, submit your request at least 7 days in advance. "
    "Employees with 5+ years of tenure qualify for 3 bonus vacation days annually."
)

if __name__ == "__main__":
    ans = annotate(answer, [policy])
    print(render_markdown(ans))
