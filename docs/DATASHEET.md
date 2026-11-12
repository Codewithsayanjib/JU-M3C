# Datasheet for JU-M3C

Following Gebru et al. (2021). A condensed version; the full datasheet is Online Resource 1 of the
paper.

**Motivation.** JU-M3C was created to measure and remove leakage between public clinical case
datasets and to provide a leak-free, multi-layer, multimodal benchmark that runs on a laptop.

**Composition.** 401,238 documents from nine public English sources (see `configs/sources.yaml`),
98.9% published case reports, connected to 737,203 figures and 15,637 MeSH descriptors. No new data
from patients; all records are drawn from openly licensed public releases.

**Collection.** Each source is read into one record format; documents are linked by exact text,
near-duplicate text, containment and PubMed Central identifier; the figure and topic layers are
attached by article identifier. No manual annotation; the duplicate detector is validated against
identifiers the sources publish (precision 99.99%).

**Splits.** Article-level train/validation/test; every published test set lies inside the JU-M3C
test split, and no article or text link crosses a split.

**Uses.** Leakage-free evaluation of clinical models; the reusable `jumc audit` tool for auditing
new datasets. Not a sample of routine hospital records.

**Distribution and licence.** Released on Zenodo under CC BY-NC-SA 4.0 upon acceptance; images as a
licence-respecting fetch index. Code: MIT.

**Maintenance.** Maintained with the paper; the audit code applies to new clinical datasets as they
are released.
