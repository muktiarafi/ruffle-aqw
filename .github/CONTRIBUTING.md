# Contributing to ruffle-aqw

This is [Ruffle](https://github.com/ruffle-rs/ruffle) with [Mugetsu](https://github.com/mugetsu-aqw/mugetsu)'s
AdventureQuest Worlds patches, kept for [Turu](https://github.com/muktiarafi/turu), which uses it
as its `ruffle/` submodule, and depended on by other projects too. Work happens on the **`aqw`**
branch; `master` mirrors upstream Ruffle and takes no pull requests here.

**Published history is never rewritten.** Other projects pin commits of this repository (a Cargo
`Cargo.lock`, a submodule), so every commit that has been on a branch or a tag here stays
reachable, and `aqw` only moves forward. See "Depending on ruffle-aqw" and "Branches and tags"
below.

Bugs in Ruffle itself belong [upstream](https://github.com/ruffle-rs/ruffle). Ruffle's own guide
is [CONTRIBUTING.md](../CONTRIBUTING.md) in the repository root; its code style applies here, its
review process doesn't.

## When to change this repository

Only when Turu needs something from the Flash player it can't do on its own: a missing Flash
API, a player hook, a fix in Ruffle's code. Everything else goes to Turu. Each change here comes
with a Turu pull request that bumps the `ruffle/` submodule to it and uses it; Turu's CI builds
and tests the result.

1. Branch from an up-to-date `aqw` (fork if you can't push here) and open the pull request
   against `aqw`.
2. Open the Turu pull request that bumps `ruffle/` to your branch, and link the two.
3. Once this one is merged, point Turu's submodule at the merged commit on `aqw`.

## Rules

- **Keep changes small and marked.** Every Turu commit has to be re-applied when `aqw` moves to
  a newer Ruffle, so touch as few upstream lines as you can, and put new code in new files where
  possible (`core/src/aqw_*.rs`).
- **Format everything.** The workspace with `cargo fmt --all`, and each crate in `patches/` with
  `rustfmt --edition <its edition> patches/<crate>/src/lib.rs` (CI's Format job). Mugetsu's
  patched files are formatted too (since 5885b04f0), so an update formats them again right after
  re-applying Mugetsu (below); Turu's commits, written on formatted code, then apply on top.
- **Patched crates** live in `patches/` and are wired up by `[patch.crates-io]` in `Cargo.toml`.
  Turu's root `Cargo.toml` repeats that section: change both.
- **Commits** follow `type: Summary` (`feat:`, `fix:`, `perf:`, `chore:`), as in the history.
  Each one should build on its own.

## Pull requests

A pull request must sit on top of the current `aqw` with no merge commits. When `aqw` moves:

```bash
git fetch origin
git rebase origin/aqw
git push --force-with-lease
```

Pull requests are merged with **Rebase and merge** only. CI ([.github/workflows/aqw.yml](workflows/aqw.yml))
must pass:

| Job | What |
|---|---|
| Rebased on aqw | the branch contains `aqw`, no merge commits |
| Format | `cargo fmt --all --check`, and rustfmt over each crate in `patches/` |
| Check | `cargo check` of the crates Turu uses, with its features, and the desktop frontend |

Ruffle's own workflows (tests, lint, web) run only for `master`. Run the tests your change
touches yourself, for example `cargo test -p ruffle_core`, and say in the pull request what you
ran.

## Depending on ruffle-aqw

Pin a **release tag**, not a branch:

```toml
# Cargo.toml: each crate you use, at the same tag
ruffle_core = { git = "https://github.com/muktiarafi/ruffle-aqw", tag = "aqw-2.0.0-1" }
```

or, for a submodule, check out the tag's commit. A tag never moves and its commit is never
deleted. Following the `aqw` branch also works (it only fast-forwards), but it moves to a new
Ruffle or Mugetsu without warning, and a new base can break your code: read the base's release
notes before taking one.

Like Turu, a dependent has to repeat this workspace's `[patch.crates-io]` section (the crates in
`patches/`) and its profiles in its own root `Cargo.toml`: Cargo reads them only from the root
workspace.

## Branches and tags

| Ref | What | Moves? |
|---|---|---|
| `aqw` | the current base, with every change on top | forward only |
| `aqw-<base>` (`aqw-1.6.0`, `aqw-2.0.0`, ...) | a base's line, frozen when the next base replaces it | forward only, for a backported fix |
| `aqw-<base>-<n>` (`aqw-2.0.0-1`, ...) | a release on that base: `n` counts from 1 per base | never |
| `master` | upstream Ruffle | upstream's |

A **base** is a Ruffle commit plus a Mugetsu version (today Ruffle `f22d29d9` + Mugetsu 2.0.0),
and its name is the Mugetsu version (`2.0.0`). A Ruffle update without a new Mugetsu starts a new
base all the same, named with a fourth number (`2.0.0.1`, then `2.0.0.2`), so no two bases share a
branch or a tag.

Maintainers tag a release after a change a dependent would want (a fix, a new hook), and with
each new base. Each release gets GitHub release notes: what changed, and for a new base the Ruffle
and Mugetsu it is on and what a dependent has to change.

Branch protection (repository settings): on `aqw` and on `aqw-*`, a pull request, no force
pushes, no deletion, and the checks above required, with admins not held to the rules. `aqw`
(branch protection) does not require a linear history, for the one merge each new base brings
(below); the `aqw-*` lines (the ruleset "Frozen bases (aqw-*)") do, and a new one may be created
before its checks have run.

## Updating to a newer Ruffle or Mugetsu

Maintainers only. Re-applying the patches gives the new base a history of its own; the old one
stays where it is, and `aqw` takes the new one with a merge that keeps it moving forward:

```bash
git fetch origin
# 1. Freeze the old base's line (it's already on `aqw`, so this only adds a name).
git push origin origin/aqw:refs/heads/aqw-<old base>
# 2. Build the new base on a local work branch (not `aqw-*`, a protected name): Mugetsu
#    re-applied on the new Ruffle and formatted in a commit of its own (Rules, "Format
#    everything"; leave out the old `style:` formatting commit), then the old line's own commits
#    on top (`git cherry-pick`, or `git rebase --onto`).
git switch -c next-base <new ruffle commit>
# ...
# 3. Join the old line in without taking any of its files: the tree stays the new base's.
git merge -s ours origin/aqw -m "chore: Move aqw to <base> (Ruffle <commit>, Mugetsu <version>)"
# 4. A fast-forward of aqw, and the new base's first release.
git push origin next-base:aqw
git tag -a aqw-<base>-1 -m "Ruffle <commit> + Mugetsu <version>"
git push origin aqw-<base>-1
```

The step 3 merge is the only merge commit `aqw` takes. Push it directly as an admin, past the
required checks and the pull request rule (a pull request would fail "Rebased on aqw", which
refuses merge commits). That is why `aqw`'s protection does not require a linear history: the CI
check keeps contributors' pull requests linear instead. Read the new line with
`git log --first-parent aqw`, and bisect with `git bisect start --first-parent`.

Open pull requests then rebase as usual (`git rebase origin/aqw`): the old line is part of the
new `aqw`, so only their own commits are replayed. Turu's
[CLAUDE.md](https://github.com/muktiarafi/turu/blob/main/CLAUDE.md) describes the steps on Turu's
side.
