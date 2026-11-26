


### Style conventions

Adhere to standard good coding practices. Also adhere to the following rules 
(which have precedence over the general rules):

- Use modular code, ideal functions are centered around
one idea, and a couple of lines up to a somehwat larger code block, don't write 
one-liner functions unless there's strong arguments (like the function
being used very often, or it contains a central assumption).
- Split code into topic-oriented "paragraphs", each 1-5 lines, or longer 
if really appropriate. Such paragraphs are headed by a comment explaining the
logic.
- Function documentation between `""" """` should be very concise. Ideally 
one line, don't list all arguments to the function with a detailed description.
If a function is more complicated, or input arguments require explanation, 
this is ideally done within a couple of lines of text that mention the 
argument and its purpose.
- Do not add type hints, docstrings, or comments to existing code that does not
  already have them, unless that is the task.
- Do not add new dependencies without explicitly asking.

### Other

- When you write tests, add them to the directory `tests/`, and include a header
you authored the test.