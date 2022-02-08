-r
How to Contribute
===================

To contribute to the codebase, users should create a pull request in the GitHub repository. To do this, first checkout the latest main branch with:

.. code:: bash

    git checkout main
    git pull


Next, you can make the code changes that you’d like to contribute.

After making your changes, please make sure to resolve all the code formatting and documentation issues that may arise by checking:

.. code:: bash

    bash scripts/lint.sh


If you see any errors or warnings as output of this, please correct them and run this script again until there are no issues anymore.
Next, we need to check if the code still passes all tests and if the code is still covered by our unit tests. For this, first run:

.. code:: bash

    python3 -m coverage run --source=vis4d -m pytest --pyargs vis4d


If the code passes all unit tests, we can now check the coverage of these unit tests by:

.. code:: bash

    python3 -m coverage report --fail-under=100 -m


The report will show you the lines that are not covered by the current unit tests. If you encounter any lines that are not covered by the unit tests, please write new unit tests for the added functionality. The tests already implemented can serve as example (all files that end with _test.py). Once you finish the unit tests, you can re-run the coverage related commands.

Once you pass all linting and coverage related checks, you can commit your changes to a new branch:

.. code:: bash

    git checkout -b my-feature
    git commit -am “awesome new feature”
    git push

Finally, you can create your pull request on the GitHub repo website and request tobiasfshr or other maintaining members as reviewer. Once your pull request is approved, it will be merged to the main branch.
