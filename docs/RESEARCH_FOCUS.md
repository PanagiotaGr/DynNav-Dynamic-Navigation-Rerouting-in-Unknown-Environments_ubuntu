# DynNav research focus

DynNav studies risk- and recoverability-aware online replanning under dynamic route invalidation.

The central research question is whether explicit recoverability estimation can reduce irreversible navigation failures without excessive path-length or computation overhead.

The primary planner objective is:

```math
J(\pi)=L(\pi)+\lambda_rR(\pi)+\lambda_{irr}I(\pi).
```

The canonical evaluation compares shortest, risk-aware, recoverability-aware and combined variants on identical dynamic bottleneck scenarios. The principal outcome is irreversible failure rate, supported by mission success, recovery success, escape-option count, cumulative risk, path overhead, replans and runtime.

The broader module collection is retained as secondary extensions. It does not define the central research claim.
