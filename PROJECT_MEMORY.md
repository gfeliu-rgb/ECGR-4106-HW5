# Project Memory: ML Homework Preferences

Use these preferences for ECGR 4106 / machine learning homework work in this repository.

## Submission Standard

- Treat the work as complete only when it is ready for a strict TA to grade, not merely scaffolded.
- The report should include the public GitHub repository link, name, student ID, course, assignment number, and clearly separated problem sections.
- If assignment wording is inconsistent, state the confusion clearly in the report or email instead of silently guessing.
- Keep the GitHub repository public and verify public visibility before submission.

## Notebooks

- Jupyter notebooks must be executed in place before submission.
- Notebooks should show outputs for code cells, tables, figures, and source-code blocks.
- Include the full problem code in the notebooks, split into readable step-by-step blocks rather than one large dump.
- Do not leave notebooks as results-only summaries; include the implementation logic needed for the grader to inspect.

## Reports

- The report must explain methods, architecture choices, hyperparameters, metrics, runtime, model size, FLOPs or complexity, and training behavior.
- Include detailed discussion and analysis, not just figures or final numbers.
- Directly answer the homework prompts for every problem.
- Explain low or unexpected results using training curves, losses, accuracy trends, and model/optimization reasoning.
- Include enough detail that the TA can see how the model was verified as learning.

## Figures and Tables

- Prefer solid-line plots with visible markers and all relevant epoch points shown.
- Avoid dotted or dashed training curves unless specifically requested.
- Include per-epoch loss and accuracy curves when training runs are part of the assignment.
- Tables should include final accuracy, losses, runtime, parameter counts, FLOPs/complexity where requested, and run status/device notes.

## Code and Reproducibility

- Use clean, documented source code in `src/` or an equivalent organized location.
- Include CPU fallback even when GPU is used for final runs.
- Keep CSV result artifacts and generated plots in the repository when they support the report.
- Provide reproducibility commands in the README or report.
- Validate notebooks for execution errors before pushing.

## Communication With TA

- Be respectful, direct, and specific.
- For late or emergency circumstances, explain the facts plainly and ask for an exception without sounding entitled.
- Mention any assignment-number confusion explicitly when relevant.
