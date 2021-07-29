# How to Contribute
Thanks for considering contributing to TF similarity!

Here is what you need to know to make a successful contribution. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Code submission guidelines

Here are the few steps you need to follow to ensure a smooth review of your
pull request:

- Ideally one PR corespond to one feature or improvement to make it easier to
  review. So **try** to split your contribution in meaning logical units.
- Your code **must** pass the unit-tests. We use `pytest` so simply run it at the root of the project.
- Your code **must** passs static analyis. We use `mypy` so simply run `mypy tensorflow_similarity/` from the root of the project.
- Your code **must** comes with unit-tests to ensure long term quality
- Your functions **must** be documented except obvious ones using the Google style.
- Your functions **must** be typed.
- Your code must pass `flake8` to ensure you follow Python coding standards.
- You **must** provide documentation (located in documentation/src) edits that document any new function/features you added.
- You **should** provide reference to research papers that describe the feature added if any.
- Please **try** to include a notebook in `notebooks/` that demonstrates the features/model if possible or extend an existing one. If you add a new notebook, make sure to provide an edit to the `README.md` that link to it.
- You **must** update the documentation/src/changelog.md that document the changes triggered by the PR.



## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
