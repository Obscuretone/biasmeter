# Contributing

Keep each commit focused on one reviewable change. Use a short, specific subject
that describes the result, such as `Handle expired peer connections`. Conventional
Commit prefixes are optional. Explain the reason or tradeoff in the body when
needed, and record the checks you ran in the pull request.

Before committing, review `git diff --cached` and run the project's documented
checks. Keep dependencies, caches, local databases, and environment files out of
Git; retain lockfiles and sanitized `.env.example` files. Squash temporary WIP,
fixup, and trial-and-error commits before merging a pull request. Keep meaningful
implementation, test, and documentation milestones when they stand on their own.

Enable the optional local checks once per clone (requires Python 3.10+):

```sh
git config --local core.hooksPath .githooks
```

The pre-commit hook checks staged paths and whitespace. The commit-msg hook rejects
empty or obviously temporary/generic subjects. CI also checks tracked paths and
new pull request commit messages. These checks catch common mistakes; they do
not replace reviewing the diff or running the project's tests.

Do not rewrite already published history during routine development. Coordinate
an exceptional cleanup with collaborators and retain a recoverable backup first.
