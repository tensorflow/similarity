import os


def abs_path(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(here, rel_path)


def read(rel_path):
    with open(abs_path(rel_path), "rt") as fd:
        return fd.read()


def write(rel_path, lines):
    with open(abs_path(rel_path), "wt") as fd:
        for line in lines:
            fd.write(line + "\n")


def increment_dev_version(previous_version):
    delim = "dev"
    if delim not in previous_version:
        raise ValueError(
            f"The previous version {previous_version} does contain a dev suffix"
        )
    # Split and increment dev version
    main_version, dev_version = previous_version.split(delim)
    dev_version = int(dev_version) + 1
    # Construct new version
    new_version = f"{main_version}{delim}{dev_version}"

    return new_version


def update_version(rel_path):
    lines = []
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            previous_version = line.split(delim)[1]
            new_version = increment_dev_version(previous_version)
            line = line.replace(previous_version, new_version)
        lines.append(line)

    write(rel_path, lines)

    return new_version


if __name__ == "__main__":
    version_path = "../tensorflow_similarity/__init__.py"

    os.system("git config --global user.email '<>'")
    os.system("git config --global user.name 'Github Actions Bot'")

    # Assumes we are in the scripts/ dir
    new_version = update_version(version_path)

    os.system(f"git add -u")
    os.system(f"git commit -m '[nightly] Increase version to {new_version}'")
    os.system("git push")
