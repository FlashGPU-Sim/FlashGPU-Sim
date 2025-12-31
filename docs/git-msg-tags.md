# Git Message Tags

This document defines the tags to be used in git commit messages.

## Tags

- `feat`: A new feature has been added.
- `sdk`: The SDK template to be modified or created.
- `bugfix`: A bug has been fixed.
- `docs`: Documentation has been updated or added.
- `test`: Test cases as well as test related infras have been added or modified.
  - DO NOT use this tag when `bugfix`, `sdk`, or `feat` is also present.
  - Use this tag only when solely modifying test cases.
- `refactor`: Code has been refactored without changing its functionality.
- `chore`: Routine tasks such as code formatting, dependency updates, etc.
- `build`: Changes to CMakeLists.txt, Makefile, or build configurations.
- `agent.<sub-tag>`: Changes related to AI agent configuration or behavior.
  - `.skill`: Modifying existing skills or adding new skills.
  - `.command`: Changes to slash commands.
  - `.settings`: Changes to agent settings or configurations.
  - `.workflow`: Changes to agent-based workflows or processes.
