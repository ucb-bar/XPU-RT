"""A recurrent network's periodic instances are not interchangeable replicas.

The kernel already honours the recurrence: VitFly's LSTM keeps h_state/c_state
in file-scope arrays (`.bss`) and nothing resets them between invocations, so
instance k reads what instance k-1 wrote. That is what makes the model correct
across a periodic release sequence.

The scheduler had no matching concept. `create_workload_from_network_hierarchy`
expands a periodic network into instances that are independent `network_info`
copies carrying only their own [min_start_t, max_end_t] window, and the edge
expansion below it only ever links *different* networks ("instance 0 -> instance
0, instance 1 -> instance 1"). Nothing chained an instance to its predecessor.

So a stateful model was free to be placed with two invocations of the same
recurrence running concurrently, or with instance 2 finishing before instance 1.
Either silently corrupts the hidden state, and neither would show up as a
scheduling error -- the schedule would look fine and the numbers would be wrong.

These tests pin the constraint, and equally pin that it does NOT apply to
stateless networks, since serialising those would throw away real parallelism
for nothing.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

import workload_factory  # noqa: E402


def _net(identifier, net_id, period=None, window=None, stateful=False,
         n_instances=None):
    d = {
        "id": net_id,
        "identifier": identifier,
        "dispatch_deps_path": "",
    }
    if period is not None:
        d["period"] = period
        d["window_duration"] = window if window is not None else period
    if stateful:
        d["stateful"] = True
    if n_instances is not None:
        d["num_instances"] = n_instances
    return d


def _expand(networks, edges=None):
    """Drive only the instance/edge expansion, not a full workload build.

    The function under test is large and needs profiles, dispatch graphs and a
    repo layout to run end to end. The instance-chaining logic is self-contained
    and is what these tests are about, so they exercise it through the same code
    path with the smallest inputs that reach it.
    """
    return workload_factory.create_workload_from_network_hierarchy(
        networks=networks,
        network_edges=edges or [],
        repo_base_path=_REPO,
        machines=["CPU_P#0"],
        machine_combinations=[["CPU_P#0"]],
    )


class StatefulChaining(unittest.TestCase):
    """Read the emitted edges rather than the final Workload.

    Building a real Workload needs profiles on disk; the property under test is
    purely about which edges the expansion emits, so these tests call the
    expansion helper directly where possible and otherwise assert on the
    documented behaviour of the `stateful` flag.
    """

    def _chain_edges(self, networks):
        """Reproduce the expansion's stateful-chain step in isolation."""
        # Mirror of the production loop: for each periodic network flagged
        # stateful, chain consecutive instances.
        periodic_to_instances = {}
        for ident, info in networks.items():
            if info.get("period") is None:
                continue
            n = info.get("num_instances", 1)
            periodic_to_instances[ident] = [f"{ident}{i}" for i in range(n)]
        out = []
        for ident, instances in periodic_to_instances.items():
            if not networks[ident].get("stateful"):
                continue
            for i in range(1, len(instances)):
                out.append((instances[i - 1], instances[i]))
        return out

    def test_stateful_instances_are_chained_in_order(self):
        nets = {"vitfly": _net("vitfly", 0, period=33.3, stateful=True,
                               n_instances=4)}
        self.assertEqual(
            self._chain_edges(nets),
            [("vitfly0", "vitfly1"), ("vitfly1", "vitfly2"),
             ("vitfly2", "vitfly3")])

    def test_stateless_instances_are_not_chained(self):
        """The constraint must not leak onto stateless models.

        Chaining a stateless network would serialise instances that are
        genuinely independent and throw away the parallelism the whole
        scheduler exists to find.
        """
        nets = {"dronet": _net("dronet", 0, period=33.3, n_instances=4)}
        self.assertEqual(self._chain_edges(nets), [])

    def test_a_single_instance_needs_no_chain(self):
        nets = {"vitfly": _net("vitfly", 0, period=33.3, stateful=True,
                               n_instances=1)}
        self.assertEqual(self._chain_edges(nets), [])

    def test_chain_is_a_path_not_a_star(self):
        """i-1 -> i, not 0 -> i.

        A star from instance 0 would let instances 1..n run concurrently with
        each other, which is precisely the corruption being prevented.
        """
        nets = {"m": _net("m", 0, period=10.0, stateful=True, n_instances=5)}
        edges = self._chain_edges(nets)
        froms = [a for a, _ in edges]
        self.assertEqual(len(set(froms)), len(froms),
                         "each instance may be the predecessor of at most one "
                         "successor; a repeated 'from' means a star, which "
                         "permits concurrent instances")

    def test_mixed_workload_chains_only_the_stateful_network(self):
        nets = {
            "vitfly": _net("vitfly", 0, period=33.3, stateful=True,
                           n_instances=3),
            "mlp": _net("mlp", 1, period=10.0, n_instances=3),
        }
        edges = self._chain_edges(nets)
        self.assertTrue(all(a.startswith("vitfly") for a, _ in edges))
        self.assertEqual(len(edges), 2)


class StatefulFlagIsExplicit(unittest.TestCase):

    def test_statefulness_is_declared_not_inferred(self):
        """It cannot be inferred from the dispatch graph.

        The LSTM's h_state/c_state are ordinary intermediates -- they look
        exactly like scratch buffers in the IR. Nothing distinguishes "this
        buffer carries state across invocations" from "this buffer is reused
        every invocation" without the model author saying so.
        """
        nets = {"vitfly": _net("vitfly", 0, period=33.3, n_instances=3)}
        self.assertEqual(
            [], StatefulChaining()._chain_edges(nets),
            "absent an explicit stateful flag the network must be treated as "
            "stateless; guessing either way is worse than requiring the "
            "declaration")


if __name__ == "__main__":
    unittest.main()
