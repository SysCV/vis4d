
# How to Contribute

To contribute to the codebase, users should create a pull request in the GitHub repository.

To do this, first checkout the latest main branch with:

```bash
git checkout main
git pull
```

Next, you can make the code changes that you’d like to contribute.

After making your changes, please make sure to resolve all the code formatting and documentation issues that may arise by checking:

```bash
bash scripts/lint.sh
```

If you see any errors or warnings as output of this, please correct them and run this script again until there are no issues anymore.

Once you pass all linting checks, you can commit your changes to a new branch:

```bash
git checkout -b my-feature
git commit -m "awesome new feature"
git push
```

Finally, you can create your pull request on the GitHub repo website and request tobiasfshr or other maintaining members as reviewer.

Once your pull request is approved, it will be merged to the main branch.
