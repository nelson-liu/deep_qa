# Models

In this module we define a number of concrete models.  The models are grouped by task, where each
task has a roughly coherent input/output specification.  See the README in each submodule for a
description of the task models in that submodule are designed to solve.

We also define a few general `Pretrainers` in a submodule here.  The `Pretrainers` in this
top-level submodule are suitable to pre-train a large class of models (e.g., any model that
encodes sentences), while more task-specific `Pretrainers` are found in that task's submodule.
