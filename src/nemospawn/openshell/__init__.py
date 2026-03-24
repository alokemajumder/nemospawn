"""OpenShell integration — NVIDIA's sandbox runtime for autonomous AI agents.

NVIDIA OpenShell is an open-source runtime that provides:
  - Gateway: control-plane API for sandbox lifecycle and auth
  - Sandbox: isolated container with kernel-level enforcement (Landlock, seccomp, netns)
  - Policy Engine: declarative YAML policies for filesystem, network, process, inference
  - Privacy Router: routes LLM API calls to controlled backends

NemoSpawn uses OpenShell sandboxes to run worker agents in isolated, policy-governed
environments with GPU passthrough and NemoSpawn coordination protocol injected.
"""
