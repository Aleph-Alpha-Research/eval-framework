# Contributions

`eval-framework` is maintained by Aleph Alpha. Wether it is an issue or a Pull Request, contributions community contributions are welcome. Please understand that we may need some time to react and give feedback.

## Commit style

* This repository uses **conventional commits:** <https://www.conventionalcommits.org/en/v1.0.0/>
* This repository has a **linear history**. I.e. no merge commits. We use rebase to merge.
* Every commit should pass the tests.


## Local setup

We use **uv** to manage Python environments and dependencies. Installation instructions for uv can be found here: <https://docs.astral.sh/uv/getting-started/installation/>

In the repository directory you can setup the environment and install dependencies using:

```shell
uv sync
```

Set up the pre-commit hooks with:

```shell
uv tool install pre-commit
uv run pre-commit install
```

## License

By contributing, you agree that your contributions will be licensed under the same license as the project. See the `LICENSE` file in the root directory for details.
