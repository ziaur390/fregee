# The guide

`mlserve-ops-guide.pdf` — a plain-language walkthrough of this whole project: what
every piece is, why it had to exist, and where you will meet the same idea again.

Written for someone who does not already know the vocabulary. Every term is defined
where it first appears and again in the glossary.

## Rebuilding it

```bash
python tasks.py guide
```

That concatenates the twelve fragments in order and prints the result to PDF with a
headless browser. On Linux or macOS, open `mlserve-ops-guide.html` and print it to
PDF — the layout is CSS print rules, so any real rendering engine will do.

A browser is required rather than a PDF library. The document depends on page-break
control, repeating table headers and print margins; a library like reportlab would
mean reimplementing all of that by hand and maintaining it.

## Why it is twelve files

One 3,300-line HTML file is unpleasant to edit and impossible to review in a diff.
The fragments are split by part, and `[01]*.html` is the load-bearing glob — the
assembler sorts and concatenates them in that order.

| File | Part |
|---|---|
| `01-cover.html` | Cover, and the shared stylesheet |
| `02-story.html` | How to read this · the whole story |
| `03-model.html` | Vocabulary · the model |
| `04-runtimes-api.html` | The three runtimes · the API |
| `05-bench-stats.html` | Benchmarking · the statistics |
| `06-drift-backup.html` | Drift detection · verified backup |
| `07-config-docker.html` | Config and tasks · Docker |
| `08-compose-monitoring.html` | Compose · monitoring |
| `09-ansible-terraform.html` | Ansible · Terraform |
| `10-ci-tests.html` | CI/CD · the test suite |
| `11-architecture-mistakes.html` | The whole picture · twenty-one mistakes |
| `12-future-glossary-cheatsheet.html` | Where you will use this · glossary · cheat sheet |

**The stylesheet lives in `01-cover.html`.** Edit it there; it applies to everything
that follows.

## A note on the numbers

Every figure in the guide is the measured value from this repository — the noise
band of 0.580–1.708, the 151.5 KiB int8 artifact, the 136,586 parameters. Where a
number was wrong and later corrected, the guide says so rather than quietly using
the right one. Part 19 is built entirely from real defects found during the work.
