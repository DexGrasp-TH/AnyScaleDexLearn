# AnyScaleDexLearn

## Purpose
Use this markdown file as the shared control document between the user and Codex.
It should support a simple cycle:
1. The user writes one task request.
2. Codex completes the task.
3. Codex writes a run log and records the completed request.
4. The user writes the next task request.

## Workspace Context
- Root workspace: `/home/mingrui/mingrui/research/project_any_scale_grasp`
- This workspace contains several repositories used by the `Any Scale Grasp` project.

## How Codex Should Use This File
- Read this file before starting work in this project area.
- Treat the content here as standing context unless a newer instruction overrides it.
- Preserve existing intent when reorganizing or extending this file.
- Keep exactly one active request in the `Next Request` section.
- Generate the request ID when starting a concrete task if the user did not provide one.
- After finishing a task, move the request into `Request History` and add a `Run Log` entry.
- Keep entries concise and factual so the file remains easy to scan.

## Current Project Description

Role 1:
- train generative models for human grasp data
- sample and visualize generated grasps
- compute statistics for generated results

Role 2:
- train a robot grasp generative model using successful grasps from the filtering stage
- sample robot grasps at inference time

## Recommended Pipeline

### Step 1: User Writes the Next Request
The user updates the `Next Request` section only.

### Step 2: Codex Executes the Task
Codex reads this file, inspects the relevant code or documents, makes the necessary changes, and verifies the result when possible.

### Step 3: Codex Records the Result
After completion, Codex:
- copies the request into `Request History`
- writes a short entry in `Run Log`
- assigns a request ID if one was not already provided
- clears or replaces `Next Request` so the file is ready for the next task

### Step 4: User Writes the Following Request
The user writes the next task in `Next Request` and the cycle repeats.

## Writing Rules
- One active request at a time.
- Request IDs should use the format `YYYY-MM-DD-XX`.
- If the user does not provide a request ID, Codex should generate the next available ID.
- Each request should have a clear success condition.
- Prefer concrete file paths, modules, repos, scripts, or experiments.
- If a task depends on prior context, reference the relevant run log entry or request ID.
- Keep historical records append-only except for obvious formatting cleanup.

## Next Request Template
Copy this structure when writing the next task:

```md
## Next Request
### Request ID
Leave blank if Codex should generate it.

### Task
Describe exactly what Codex should do.

### Goal
Describe the expected result or success condition.

### Constraints
- List constraints here.

### Relevant Paths
- Add files, directories, repos, or documents here.

### Notes
- Add supporting context here.
```

## Next Request

### Request ID
Leave blank if Codex should generate it.

### Task
Describe exactly what Codex should do.

### Goal
Describe the expected result or success condition.

### Constraints
- List constraints here.

### Relevant Paths
- Add files, directories, repos, or documents here.

### Notes
- Add supporting context here.



## Request History

### Request ID
2026-04-28-01

### Task
Check whether `traj_length: 3` and `joint_num: 16` in `AnyScaleDexLearn/dexlearn/config/data/humanMulti.yaml` are actually used by the code. Remove them from configs only if they are unused.

### Goal
Confirm whether the parameters are live config inputs and clean them up only if they are dead.

### Constraints
- Keep the config unchanged if the parameters are still referenced.

### Relevant Paths
- `AnyScaleDexLearn/dexlearn/config/data/humanMulti.yaml`
- `AnyScaleDexLearn/dexlearn/config/algo/humanMultiDiffusion.yaml`
- `AnyScaleDexLearn/dexlearn/config/algo/humanMultiHierar.yaml`
- `AnyScaleDexLearn/dexlearn/network/final_layers/diffusion.py`

### Notes
- `joint_num` and `traj_length` are still used through Hydra interpolation from `data.*` into the human multi model heads, so the keys were kept and only the misleading `# unused?` comments were removed.

## Run Log

- `2026-04-28-01`: Verified that `data.joint_num` and `data.traj_length` from `humanMulti.yaml` are consumed by `humanMultiDiffusion.yaml` and `humanMultiHierar.yaml`, and then used in `dexlearn/network/final_layers/diffusion.py`. Kept both keys and removed the misleading `# unused?` comments.

## After Each Task
Codex should update these sections in order:
1. complete the requested work
2. append the finished request to `Request History`
3. append a concise entry to `Run Log`
4. replace `Next Request` with the next user-written request or leave a blank template
