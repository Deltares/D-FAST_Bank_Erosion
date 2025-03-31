# Docstrings and Doctest

Proper documentation is essential for code readability and maintainability. This guide explains how to write docstrings using the Google style and test them using `doctest`.

---

## 1. Writing Docstrings in Google Style
Google style docstrings are a clean and readable way to document your code. Here is a breakdown of the format:

### Example Docstring
```python
def add_numbers(a: int, b: int) -> int:
    """
    Add two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of the two numbers.

    Examples:
        >>> add_numbers(1, 2)
        3
        >>> add_numbers(-1, 5)
        4
    """
    return a + b
```

### Key Sections of a Google Style Docstring
1. **Summary**: A short description of the function.
2. **Args**: A list of all parameters with their types and descriptions.
3. **Returns**: A description of the return value(s).
4. **Raises** (optional): A list of exceptions the function may raise.
5. **Examples**: Code examples demonstrating how to use the function.

### Additional Notes
- Use `"""` for multi-line docstrings.
- Align descriptions for readability.
- Keep the summary concise and to the point.

---

## 2. Testing Docstrings with Doctest
`doctest` allows you to test the examples provided in your docstrings.

### Running Doctests
1. Save your code with properly formatted docstrings.
2. Use the following command to run `doctest`:

   ```bash
   python -m doctest -v your_script.py
   ```

   The `-v` flag provides verbose output, showing which tests passed or failed.

### Example Script with Doctest
```python
def multiply_numbers(a: int, b: int) -> int:
    """
    Multiply two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The product of the two numbers.

    Examples:
        >>> multiply_numbers(2, 3)
        6
        >>> multiply_numbers(-1, 5)
        -5
        >>> multiply_numbers(0, 10)
        0
    """
    return a * b

```

### Common Commands
- **Run all tests**: Run the script with doctest as shown above.
- **Check specific failures**: Look at the detailed output to understand why a test failed.

---

## 3. Debugging and Troubleshooting
1. Ensure your examples in the docstring match the actual output exactly (including whitespace).
2. Use the `# doctest: +SKIP` directive to skip examples that should not be tested:

   ```python
   >>> some_function()  # doctest: +SKIP
   ```

3. If you encounter issues with floating-point numbers, use the `# doctest: +ELLIPSIS` directive to allow partial matching:

   ```python
   >>> divide_numbers(1, 3)
   0.333...  # doctest: +ELLIPSIS
   ```

---

## 4. Best Practices
- Write examples for edge cases (e.g., zero, negative numbers, large inputs).
- Ensure all public functions, methods, and classes have docstrings.
- Regularly run `doctest` to ensure your examples remain up to date.

---

## 5. Additional Resources
- [Google Python Style Guide: Docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
- [Python Doctest Documentation](https://docs.python.org/3/library/doctest.html)

By following these guidelines, you can ensure your code is both well-documented and well-tested. Thank you for contributing!
