import re 
import inspect
pattern = re.compile(r"^Hello")

print(pattern.match("Hello World")) # This will match
print(pattern.match("Hi there")) # This will not match

print(re.match(r"\d","123abc")) # This will match

"""r"^Hello$
^  : start here
Hello : match Hello
$ : end here
"""

# print(re.match(r"^Hello$","Hello")) # This will match
# print(re.match(r"^Hello$","Hello world")) # This will not match
# pattern = r"^a+"
# print(re.match(pattern,"aaab"))
# second_pattern = r"dog|cat"
# print(re.match(second_pattern,"cat dog"))

_PARAM = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Returns?|Raises?|Yields?|Examples?|Notes?)\s*:\s*$", re.I
)

docs  = """
Calculate the area of a rectangle.
This function multiplies width and height.

Args:
    width (float): Width of the rectangle.
        Must be greater than zero.
    height (float): Height of the rectangle.

Returns:
    float: The calculated area.

Raises:
    ValueError: If dimensions are negative.
"""

text = inspect.cleandoc(docs).splitlines()
#print(text)
# for line in text:
#     print(line.strip())
state ,current , = "summary",None
for line in inspect.cleandoc(docs).splitlines():
    section = _SECTION.match(line)
    if section:
        state = (
            "args"
            if section.group(1).lower()
            in ("args", "arguments", "parameters")
            else "other"
        )
