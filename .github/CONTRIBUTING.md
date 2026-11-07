# Contributing to ruffle-aqw

This is [Ruffle](https://github.com/ruffle-rs/ruffle) with [Mugetsu](https://github.com/mugetsu-aqw/mugetsu)'s
AdventureQuest Worlds patches, kept for [Turu](https://github.com/muktiarafi/turu), which uses it
as its `ruffle/` submodule. Work happens on the **`aqw`** branch; `master` mirrors upstream
Ruffle and takes no pull requests here.

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
- **Don't reformat.** Mugetsu's patched files aren't rustfmt-clean, and reformatting them makes
  every update conflict. Format only the lines you write.
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
| Check | `cargo check` of the crates Turu uses, with its features, and the desktop frontend |

Ruffle's own workflows (tests, lint, web) run only for `master`. Run the tests your change
touches yourself, for example `cargo test -p ruffle_core`, and say in the pull request what you
ran.

## Updating to a newer Ruffle or Mugetsu

Maintainers only: Mugetsu is re-applied on the new Ruffle, then Turu's commits on top, and `aqw`
is replaced (a force-push) and tagged. Open pull requests then need a rebase onto the new `aqw`.
Turu's [CLAUDE.md](https://github.com/muktiarafi/turu/blob/main/CLAUDE.md) describes the steps on
Turu's side.
