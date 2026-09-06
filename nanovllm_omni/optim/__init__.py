"""Performance + optimization utilities (TK-011..).

Leaf-only facade: do NOT re-export ``cuda_graph`` / ``talker_cuda_graph`` at
the top level. ``cuda_graph`` imports ``models.minimind_omni._sampling`` at
module scope; re-exporting here would pull the whole models package in when
``import nanovllm_omni.optim`` runs, breaking the leaf-import contract.
"""
