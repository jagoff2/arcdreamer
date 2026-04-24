"""Grounded internal language for hypotheses, questions, and memory keys.

This is not a chatbot or hidden prompt.  It is a compact controlled vocabulary
whose tokens are grounded in observations, transition evidence, and rule objects.
The planner consumes the same tokens for retrieval and diagnostic action scoring.
"""

from __future__ import annotations

from .hypotheses import Hypothesis, HypothesisEngine
from .types import ActionName, StructuredState, action_family


class GroundedLanguage:
    def state_tokens(self, state: StructuredState) -> tuple[str, ...]:
        tokens: list[str] = [
            f"grid={state.grid.shape[0]}x{state.grid.shape[1]}",
            f"objects={len(state.objects)}",
            f"dominant=c{state.dominant_color}",
        ]
        for obj in state.objects[:24]:
            tokens.append(f"obj:c{obj.color}:area{min(obj.area, 20)}")
            for tag in obj.role_tags[:3]:
                tokens.append(f"role:{tag}:c{obj.color}")
        for rel in state.relations[:32]:
            tokens.append(f"rel:{rel.relation}")
        return tuple(tokens)

    def hypothesis_tokens(self, hypotheses: tuple[Hypothesis, ...]) -> tuple[str, ...]:
        tokens: list[str] = []
        for hyp in hypotheses:
            tokens.extend(hyp.as_tokens())
        return tuple(tokens[:96])

    def memory_tokens(self, state: StructuredState, engine: HypothesisEngine) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.state_tokens(state), *self.hypothesis_tokens(engine.credible_hypotheses(limit=8)))))

    def belief_sentences(self, engine: HypothesisEngine, *, limit: int = 8) -> tuple[str, ...]:
        sentences: list[str] = []
        for hyp in engine.credible_hypotheses(limit=limit):
            p = round(hyp.posterior, 2)
            if hyp.kind == "action_moves_object":
                sentences.append(
                    f"belief p={p}: action {hyp.action_family} moves object {hyp.params.get('object_signature')} by {hyp.params.get('delta')}"
                )
            elif hyp.kind.startswith("reward"):
                sentences.append(f"belief p={p}: progress may depend on color c{hyp.params.get('color')}")
            elif hyp.kind.startswith("mode"):
                sentences.append(f"belief p={p}: action {hyp.action_family} may change latent mode")
            else:
                sentences.append(f"belief p={p}: {hyp.description}")
        return tuple(sentences)

    def questions(self, engine: HypothesisEngine, *, limit: int = 6) -> tuple[str, ...]:
        questions: list[str] = []
        for hyp in engine.uncertain_hypotheses(limit=limit):
            p = round(hyp.posterior, 2)
            if hyp.kind == "action_moves_object":
                questions.append(f"question p={p}: test whether {hyp.action_family} consistently causes delta {hyp.params.get('delta')}")
            elif hyp.kind == "targeted_action_changes_object":
                questions.append(f"question p={p}: test targeted {hyp.action_family} on color c{hyp.params.get('color')}")
            elif hyp.kind.startswith("reward"):
                questions.append(f"question p={p}: test whether color c{hyp.params.get('color')} predicts progress")
            else:
                questions.append(f"question p={p}: falsify {hyp.kind} via action {hyp.action_family}")
        return tuple(questions)

    def plan_sentence(self, action: ActionName, *, components: dict[str, float], reason: str) -> str:
        parts = [f"plan: do {action_family(action)}"]
        if reason:
            parts.append(f"because {reason}")
        for key in ("expected_reward", "information_gain", "novelty", "world_uncertainty"):
            if key in components:
                parts.append(f"{key}={components[key]:.3f}")
        return "; ".join(parts)
