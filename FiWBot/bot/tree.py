"""
The exact public betting tree of the game (chip-accurate), built by
replaying game.py's own rules on a dummy deal. Every node is identified by
its action sequence, so mapping a live hand to a node is exact.

Node arrays (index = node id):
    actor[n]      0 = button (small blind), 1 = big blind, -1 = terminal
    street[n]     0..3
    kind[n]       0 = decision, 1 = fold terminal, 2 = showdown terminal
    children[n]   dict action -> node id
    contrib[n]    (button chips in, big blind chips in) at terminals
    folder[n]     who folded (terminal kind 1)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Action, Deal, Game  # noqa: E402

_DUMMY = Deal(hole_cards=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), board=(10, 11, 12, 13, 14))


class Tree:
    def __init__(self):
        self.actor, self.street, self.kind = [], [], []
        self.children, self.contrib, self.folder = [], [], []
        self.parent, self.key = [], []
        self.index = {}
        self._build()

    def _new(self, g: Game, key: tuple, parent: int) -> int:
        n = len(self.actor)
        self.index[key] = n
        self.key.append(key)
        self.parent.append(parent)
        self.children.append({})
        if g.is_terminal:
            res = g.result()
            self.actor.append(-1)
            self.street.append(int(g.street))
            self.kind.append(2 if res.went_to_showdown else 1)
            self.contrib.append((g.contributed[0], g.contributed[1]))
            self.folder.append(-1 if res.went_to_showdown else (1 - res.winner_seat))
        else:
            self.actor.append(g.to_act)
            self.street.append(int(g.street))
            self.kind.append(0)
            self.contrib.append((g.contributed[0], g.contributed[1]))
            self.folder.append(-1)
        return n

    def _build(self):
        root = Game(_DUMMY, button_seat=0)
        stack = [(root, (), -1)]
        while stack:
            g, key, parent = stack.pop()
            n = self._new(g, key, parent)
            if parent >= 0:
                self.children[parent][key[-1]] = n
            if g.is_terminal:
                continue
            for a in g.legal_actions():
                c = g.clone()
                c.apply(a)
                stack.append((c, key + (int(a),), n))

    def __len__(self):
        return len(self.actor)


if __name__ == "__main__":
    t = Tree()
    import collections
    dec = sum(1 for k in t.kind if k == 0)
    print("nodes", len(t), "decision", dec, "fold-terminals", t.kind.count(1), "showdowns", t.kind.count(2))
    print("decision nodes per street", collections.Counter(s for s, k in zip(t.street, t.kind) if k == 0))
