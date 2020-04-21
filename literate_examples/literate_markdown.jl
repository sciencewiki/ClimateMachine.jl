# How to generate a literate file

using Literate
const examples_directory = pwd() * "/literate_examples/"
const output_directory = pwd() * "/literate_examples/"

# list of examples
examples = readdir(examples_directory)

for example in examples
    example_filepath = examples_directory * example
    Literate.markdown(example_filepath, output_directory, documenter = true)
end
