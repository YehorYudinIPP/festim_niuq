# Contributing to FESTIM-NIUQ

Thank you for your interest in contributing to FESTIM-NIUQ! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Issues

If you find a bug, have a feature request, or have a question about the software, please [open an issue](https://github.com/YehorYudinIPP/festim_niuq/issues/new) on GitHub. When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behaviour
- Your operating system, Python version, and relevant package versions
- Any error messages or tracebacks

### Suggesting Enhancements

Enhancement suggestions are welcome. Please open an issue describing:

- The problem or limitation you are addressing
- Your proposed solution or feature
- Why this would be useful to other users

### Submitting Pull Requests

1. **Fork** the repository and create a new branch from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**, following the coding style of the existing codebase.

3. **Add or update tests** for your changes where applicable.

4. **Run the test suite** to ensure nothing is broken:
   ```bash
   pytest tests/
   ```

5. **Commit** your changes with a clear, descriptive commit message.

6. **Push** to your fork and [submit a pull request](https://github.com/YehorYudinIPP/festim_niuq/compare).

### Coding Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code
- Use descriptive variable and function names
- Add docstrings to all public functions, classes, and methods
- Keep functions focused and modular

### Testing

- Write unit tests for new functionality
- Use `pytest` as the testing framework
- Mock external dependencies (FESTIM, EasyVVUQ) where appropriate
- Ensure all existing tests pass before submitting a PR

## Getting Help

If you need help or have questions:

- Open an issue on the [GitHub issue tracker](https://github.com/YehorYudinIPP/festim_niuq/issues)
- Check existing issues for similar questions
- Refer to the [README](README.md) for installation and usage instructions

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behaviour via the issue tracker.
