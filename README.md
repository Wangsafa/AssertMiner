# AssertMiner
This repository contains the source code for the paper *"AssertMiner: Module-Level Spec Generation and Assertion Mining using Static Analysis Guided LLMs"*.

## Files

- **File 1: `structural analysis.py`**  
  Takes RTL code and the original design specification (converted into Markdown format) as input, and outputs the extracted relevant structural information.

- **File 2: `spec extraction.txt`**  
  Takes the extracted structural information and the design architecture description (in Markdown format) as input. Using the content of this prompt file, the large model is guided to output the specification file describing the functionality of the submodules in the design.
  
- **File 3: item extraction.txt`**  
  Using the content of this prompt file, the large model is guided to extract detailed descriptions of verification points from the generated submodule specification file.

- **File 4: `assertion generation.txt`**  
  Takes the structural information, the original Markdown specification file, the generated submodule specification file, the descriptions of verification points, and relevant module input/output interfaces along with RTL code snippets of module instantiations as input. Using the content of this prompt file, the large model is guided to generate deep assertions.

- **File 5: `result_log`**  
  Contains logs preserved during our experiments.

**Note:** In our experiments, we found that the accuracy varies considerably across different submodules.
