# Repository workflow

User decision, 2026-11-05: work directly on `main` in the original checkout.
Do not create feature branches or additional worktrees. Preserve existing work
when consolidating history. Delete stale branches only after confirming their
commits are contained in `main`. Releases are created from explicit version tags
on `main`; merging work does not itself authorize a new release tag.
