import itertools
import pathlib
import random

import numpy as np

_State = int
_Label = str
_StateTransitions = dict[_Label, _State]

END_OF_SEQUENCE = "#"


class DFA:
    def __init__(
        self,
        transitions: dict[_State, _StateTransitions],
        accepting_states=set[_State],
    ):
        self._states: set[_State] = set(transitions.keys()) | set(
            itertools.chain(*[tuple(x.values()) for x in transitions.values()])
        )
        self._transitions: dict[_State, _StateTransitions] = transitions
        self._accepting_states: set[_State] = accepting_states

    def generate_string(self):
        curr_state = 0
        string = ""
        while curr_state not in self._accepting_states:
            char, curr_state = random.choice(
                tuple(self._transitions[curr_state].items())
            )
            string += char
        return string

    def visualize(self, name: str):
        dot = "digraph G {\ncolorscheme=X11\n"
        # inputs and outputs
        for state in sorted(self._states):
            if state in self._accepting_states:
                style = "peripheries=2"
            else:
                style = ""
            description = f'[label="q{state}" {style}]'

            dot += f"{state} {description}\n"

        for state, transitions in self._transitions.items():
            for label, neighbor in transitions.items():
                dot += f'{state} -> {neighbor} [ label="{label}" ];\n'

        dot += "}"

        path = pathlib.Path(f"dot_files/dfa_{name}.dot")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(dot)

    def get_optimal_data_given_grammar_for_dfa(self, corpus) -> float:
        curr_state = 0
        optimal_d_g_per_sequence = []

        for b in range(corpus.target_sequence.shape[0]):
            curr_sequence_optimal_d_g = 0.0
            for i in range(corpus.target_sequence.shape[1]):
                curr_transitions = self._transitions[curr_state]
                curr_sequence_optimal_d_g += np.log2(len(curr_transitions))

                curr_target_vector = corpus.target_sequence[b, i]

                if curr_target_vector.shape[0] == 1:
                    curr_input = curr_target_vector[0]
                else:
                    curr_input = curr_target_vector.argmax()

                curr_char = corpus.vocabulary[curr_input]
                curr_state = curr_transitions[curr_char]

                if curr_state in self._accepting_states:
                    optimal_d_g_per_sequence.append(curr_sequence_optimal_d_g)
                    curr_state = 0
                    break

        return float(np.dot(corpus.sample_weights, optimal_d_g_per_sequence))
