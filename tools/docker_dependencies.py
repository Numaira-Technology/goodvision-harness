"""Emit only the dependency groups used by one Docker image."""

import argparse
from pathlib import Path

import tomllib


def requirements(project, extras=()):
    values = list(project["project"]["dependencies"])
    for extra in extras:
        values.extend(project["project"]["optional-dependencies"][extra])
    values.extend(project["build-system"]["requires"])
    return "".join(value + "\n" for value in dict.fromkeys(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--extras", nargs="*", default=[])
    args = parser.parse_args()
    data = tomllib.loads(args.project.read_text())
    args.output.write_text(requirements(data, args.extras))


if __name__ == "__main__":
    main()
