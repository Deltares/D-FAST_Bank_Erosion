# Pre-Commit Hooks

The **pre-commit hooks** tool is integrated into this repository to automate code quality checks, formatting, and validation before committing changes. These hooks enforce consistency and ensure that only well-validated code is committed, improving overall codebase quality.

## **Features of Pre-Commit Hooks**
- **Automated Validation:** Validates file formats, YAML/JSON structure, and checks for large files or trailing whitespace.
- **Code Formatting:** Automatically formats Python code using `black` and organizes imports with `isort`.
- **Static Analysis:** Enforces linting rules using `flake8` and additional plugins for advanced checks.
- **Security Checks:** Detects private keys and merge conflicts in committed code.
- **Test Integration:** Runs `pytest` for code and documentation tests, ensuring no regression in functionality.

## **Configured Hooks**
Some of the key hooks configured for this repository include:
- **File Formatting and Cleanup:**
    - Trim trailing whitespace.
    - Fix file encodings.
    - Validate YAML and JSON files.
- **Code Linting and Formatting:**
    - `black` for consistent Python code style.
    - `isort` for sorting imports.
    - `flake8` for linting.
- **Testing:**
    - Runs unit tests and checks coverage using `pytest`.
    - Validates Jupyter notebooks (`nbval`).
    - Ensures all Python docstrings follow conventions (`doctest`).
- **Git and Metadata Checks:**
    - Ensures commits are not directly made to protected branches (e.g., `main`).
    - Verifies commit messages are well-formed.

## How to Use Pre-Commit Hooks

Pre-commit hooks are an excellent way to ensure code quality and consistency before committing changes to the repository.
This guide explains how to set up and use pre-commit hooks for this repository.

### 1. Install Pre-Commit
Before you begin, ensure you have Python installed on your system. all the necessary packages for pre-commit hooks to
work are listed as a separate dependency group in the pyproject.toml `pre-commit`. Follow these
steps to install pre-commit:

```bash
poetry install --with pre-commit
```

### 2. Install Git and Clone the Repository
Ensure that Git is installed on your system. Then, clone the repository:

```bash
git clone <repository_url>
cd <repository_name>
```

### 3. Install Pre-Commit Hooks
To set up the pre-commit hooks defined in the repository's configuration file:

```bash
pre-commit install
```

This command installs the hooks so they run automatically every time you create a commit.

### 4. Run Pre-Commit Hooks Manually (Optional)
You can also run the pre-commit hooks manually to test your changes before committing:

```bash
pre-commit run --all-files
```

This will run all the hooks against the files in the repository.

### 5. Configure the Pre-Commit Hooks
The repository includes a `.pre-commit-config.yaml` file, which defines the hooks to be executed. If you need to modify the hooks, edit this file and reinstall the hooks:

```bash
pre-commit install
```

#### Example `.pre-commit-config.yaml` File
Here is an example configuration file for common hooks:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0  # Use the latest stable version
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

### 6. Debugging and Troubleshooting
If you encounter issues with pre-commit hooks:

1. Ensure all dependencies required by the hooks are installed.
2. Use the `--verbose` flag to get detailed output:

   ```bash
   pre-commit run --all-files --verbose
   ```
3. Check the official documentation for each specific hook if errors persist.

### 7. Best Practices
- Always run the hooks manually if you suspect your changes may not comply with repository standards.
- Avoid skipping hooks unless absolutely necessary. If skipping is required, use the following command to bypass hooks:

  ```bash
  git commit --no-verify
  ```

- Periodically update the hooks by running:

  ```bash
  pre-commit autoupdate
  ```

### 8. Uninstall Hooks:
- To remove pre-commit hooks from your repository, run:

   ```bash
   pre-commit uninstall
   ```
   This will remove the hooks from the repository.

### 9. Additional Resources
- [Pre-Commit Documentation](https://pre-commit.com/)
- [Available Hooks](https://pre-commit.com/hooks.html)

---
By following these guidelines, you'll help maintain a clean, consistent, and high-quality codebase. Thank you for contributing!
