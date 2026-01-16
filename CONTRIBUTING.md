# Contributing to Fibonacci KV Cache

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/fibonacci-kv-cache/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and environment details
   - Code samples if applicable

### Suggesting Features

1. Check existing issues and discussions
2. Create a new issue with:
   - Clear use case description
   - Proposed API or behavior
   - Why this would be valuable

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Run linting: `black . && flake8 && mypy fibonacci_kv_cache`
7. Commit with clear messages
8. Push to your fork
9. Create a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/fibonacci-kv-cache.git
cd fibonacci-kv-cache

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black fibonacci_kv_cache tests examples
flake8 fibonacci_kv_cache tests examples
mypy fibonacci_kv_cache
```

## Testing Guidelines

- Write tests for all new functionality
- Maintain or improve code coverage
- Use property-based tests for algorithmic correctness
- Include both unit tests and integration tests

## Code Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use type hints for all functions
- Write docstrings for all public APIs
- Keep functions focused and small

## Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Be descriptive but concise
- Reference issues: "Fix #123: Description"

## Questions?

Open a discussion or reach out to the maintainers.

Thank you for contributing!
