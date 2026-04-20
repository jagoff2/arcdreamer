from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import math

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import ActionName, ObjectState, Relation, StructuredState

_DYNAMIC_TAG_PREFIXES = ("repaired_",)
_ROLE_TAGS = ("repaired_mover", "repaired_goal", "repaired_control")

MatchSignature = tuple[int, tuple[str, ...]]


@dataclass
class RepairEvidence:
    support: int = 0
    contradiction: int = 0

    @property
    def confidence(self) -> float:
        return float(self.support + 1) / float(self.support + self.contradiction + 2)

    @property
    def balance(self) -> int:
        return self.support - self.contradiction

    def observe(self, *, supported: bool, weight: int = 1) -> None:
        if supported:
            self.support += max(int(weight), 1)
        else:
            self.contradiction += max(int(weight), 1)


@dataclass
class TrackRoleState:
    mover: RepairEvidence = field(default_factory=RepairEvidence)
    goal: RepairEvidence = field(default_factory=RepairEvidence)
    control: RepairEvidence = field(default_factory=RepairEvidence)


class OnlineLinearRepairModel:
    def __init__(self) -> None:
        self.learning_rate = 1.15
        self.weights: dict[str, list[float]] = {
            "split": [-0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
            "merge": [-0.65, 0.0, 0.0, 0.0, 0.0, 0.0],
            "rebind": [-0.4, 0.0, 0.0, 0.0, 0.0],
        }

    def reset(self) -> None:
        self.__init__()

    def score(self, repair_type: str, features: tuple[float, ...]) -> float:
        weights = self.weights[repair_type]
        margin = sum(weight * feature for weight, feature in zip(weights, features, strict=False))
        return _sigmoid(margin)

    def update(self, repair_type: str, features: tuple[float, ...], label: float, *, weight: float = 1.0) -> None:
        prediction = self.score(repair_type, features)
        error = max(min(label - prediction, 1.0), -1.0)
        weights = self.weights[repair_type]
        scale = self.learning_rate * max(weight, 0.1)
        for index, feature in enumerate(features):
            weights[index] += scale * error * feature


class RepresentationRepairWorkspace:
    def __init__(self) -> None:
        self.split_hypotheses: dict[MatchSignature, RepairEvidence] = {}
        self.merge_hypotheses: dict[MatchSignature, RepairEvidence] = {}
        self.rebind_hypotheses: dict[MatchSignature, RepairEvidence] = {}
        self.track_roles: dict[str, TrackRoleState] = {}
        self.proposal_model = OnlineLinearRepairModel()
        self.last_state: StructuredState | None = None
        self.next_track_index = 0

    def reset(self) -> None:
        self.split_hypotheses.clear()
        self.merge_hypotheses.clear()
        self.rebind_hypotheses.clear()
        self.track_roles.clear()
        self.proposal_model.reset()
        self.last_state = None
        self.next_track_index = 0

    def augment(self, state: StructuredState, *, commit: bool = True) -> StructuredState:
        objects = list(state.objects)
        objects = self._apply_merge_repairs(objects)
        objects = self._apply_split_repairs(objects)
        objects = self._bind_track_ids(objects)
        objects = self._apply_role_labels(objects)
        objects = sorted(objects, key=_sort_key)
        repaired = replace(
            state,
            objects=tuple(objects),
            relations=tuple(_build_relations(tuple(objects))),
        )
        if commit:
            self.last_state = repaired
        return repaired

    def observe_transition(
        self,
        before: StructuredState,
        action: ActionName,
        reward: float,
        after: StructuredState,
        *,
        terminated: bool,
    ) -> None:
        del terminated
        before_groups = _group_by_match_signature(before.objects)
        after_groups = _group_by_match_signature(after.objects)
        for signature in before_groups.keys() | after_groups.keys():
            before_group = before_groups.get(signature, [])
            after_group = after_groups.get(signature, [])
            before_count = len(before_group)
            after_count = len(after_group)
            split = self.split_hypotheses.setdefault(signature, RepairEvidence())
            merge = self.merge_hypotheses.setdefault(signature, RepairEvidence())
            if after_count > before_count:
                split.observe(supported=True, weight=after_count - before_count)
            elif before_count > 0:
                split.observe(supported=False)
            if before_count > after_count:
                merge.observe(supported=True, weight=before_count - after_count)
            elif after_count > 0:
                merge.observe(supported=False)
            if before_count == 1:
                split_features = _split_feature_vector(before_group[0])
                if split_features is not None:
                    self.proposal_model.update(
                        "split",
                        split_features,
                        1.0 if after_count > before_count else 0.0,
                        weight=float(max(abs(after_count - before_count), 1)),
                    )
            if before_count >= 2:
                merge_features = _merge_feature_vector(before_group)
                self.proposal_model.update(
                    "merge",
                    merge_features,
                    1.0 if before_count > after_count else 0.0,
                    weight=float(max(abs(before_count - after_count), 1)),
                )
            if min(before_count, after_count) >= 2:
                sorted_cost = _ordered_group_cost(before_group, after_group)
                nearest_cost = _nearest_group_cost(before_group, after_group)
                rebind = self.rebind_hypotheses.setdefault(signature, RepairEvidence())
                rebind.observe(supported=(sorted_cost - nearest_cost) >= 1.5)
                self.proposal_model.update(
                    "rebind",
                    _rebind_feature_vector(before_group, after_group, sorted_cost, nearest_cost),
                    1.0 if (sorted_cost - nearest_cost) >= 1.5 else 0.0,
                    weight=max(sorted_cost - nearest_cost, 1.0),
                )

        matched_ids = {obj.object_id for obj in before.objects} & {obj.object_id for obj in after.objects}
        changed_track_ids: set[str] = set()
        for object_id in matched_ids:
            before_obj = next(obj for obj in before.objects if obj.object_id == object_id)
            after_obj = next(obj for obj in after.objects if obj.object_id == object_id)
            if _object_motion(before_obj, after_obj) > 0.1:
                changed_track_ids.add(object_id)

        context = build_action_schema_context(before.affordances, dict(before.action_roles))
        schema = build_action_schema(action, context)
        action_type = schema.action_type
        state_delta = _state_delta(before, after)
        goal_activated = _goal_activated(before, after)

        for object_id in matched_ids:
            role_state = self.track_roles.setdefault(object_id, TrackRoleState())
            if action_type == "move":
                role_state.mover.observe(supported=object_id in changed_track_ids)
        interaction_target = _interaction_target_object(before, action, context)
        if interaction_target is not None:
            role_state = self.track_roles.setdefault(interaction_target.object_id, TrackRoleState())
            role_state.control.observe(
                supported=(state_delta >= 0.05 or abs(float(reward)) > 0.01 or goal_activated),
            )

        if reward > 0.0 or goal_activated:
            reward_targets = _positive_reward_targets(before, after)
            for object_id in reward_targets:
                role_state = self.track_roles.setdefault(object_id, TrackRoleState())
                role_state.goal.observe(supported=True)

    def _apply_merge_repairs(self, objects: list[ObjectState]) -> list[ObjectState]:
        grouped = _group_by_match_signature(tuple(objects))
        consumed: set[str] = set()
        merged_objects: list[ObjectState] = []
        for signature, group in grouped.items():
            if len(group) < 2:
                continue
            adjacency: dict[str, set[str]] = {obj.object_id: set() for obj in group}
            by_id = {obj.object_id: obj for obj in group}
            for index, source in enumerate(group):
                for target in group[index + 1 :]:
                    if _merge_candidate(source, target):
                        adjacency[source.object_id].add(target.object_id)
                        adjacency[target.object_id].add(source.object_id)
            visited: set[str] = set()
            for obj in group:
                if obj.object_id in visited:
                    continue
                component = []
                frontier = [obj.object_id]
                visited.add(obj.object_id)
                while frontier:
                    current = frontier.pop()
                    component.append(by_id[current])
                    for neighbor in adjacency[current]:
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        frontier.append(neighbor)
                component_evidence = self.merge_hypotheses.get(signature)
                proposal_score = self._proposal_score(
                    "merge",
                    _merge_feature_vector(component),
                    component_evidence,
                )
                if len(component) > 1 and proposal_score >= 0.62:
                    consumed.update(item.object_id for item in component)
                    merged_objects.append(_merge_objects(component))
        if not merged_objects:
            return objects
        return [obj for obj in objects if obj.object_id not in consumed] + merged_objects

    def _apply_split_repairs(self, objects: list[ObjectState]) -> list[ObjectState]:
        repaired: list[ObjectState] = []
        for obj in objects:
            signature = _match_signature(obj)
            evidence = self.split_hypotheses.get(signature)
            split_features = _split_feature_vector(obj)
            if split_features is None or obj.area < 4:
                repaired.append(obj)
                continue
            split_objects = _split_object(obj)
            if len(split_objects) <= 1 or self._proposal_score("split", split_features, evidence) < 0.64:
                repaired.append(obj)
            else:
                repaired.extend(split_objects)
        return repaired

    def _bind_track_ids(self, objects: list[ObjectState]) -> list[ObjectState]:
        if self.last_state is None:
            bound = []
            for obj in objects:
                bound.append(replace(obj, object_id=self._new_track_id()))
            return bound
        previous_groups = _group_by_match_signature(self.last_state.objects)
        current_groups = _group_by_match_signature(tuple(objects))
        bound_by_temp_id: dict[str, ObjectState] = {}
        assigned_previous: set[str] = set()
        for signature, current_group in current_groups.items():
            previous_group = list(previous_groups.get(signature, []))
            unmatched_current = list(current_group)
            use_learned_rebind = False
            if len(previous_group) >= 2 and len(unmatched_current) >= 2 and len(previous_group) == len(unmatched_current):
                sorted_cost = _ordered_group_cost(previous_group, unmatched_current)
                nearest_cost = _nearest_group_cost(previous_group, unmatched_current)
                use_learned_rebind = self._proposal_score(
                    "rebind",
                    _rebind_feature_vector(previous_group, unmatched_current, sorted_cost, nearest_cost),
                    self.rebind_hypotheses.get(signature),
                ) >= 0.58
            if use_learned_rebind:
                while previous_group and unmatched_current:
                    best_pair = min(
                        (
                            (_object_cost(previous, current), previous, current)
                            for previous in previous_group
                            for current in unmatched_current
                        ),
                        key=lambda item: item[0],
                    )
                    _, previous, current = best_pair
                    assigned_previous.add(previous.object_id)
                    bound_by_temp_id[current.object_id] = replace(current, object_id=previous.object_id)
                    previous_group.remove(previous)
                    unmatched_current.remove(current)
            else:
                pair_count = min(len(previous_group), len(unmatched_current))
                for index in range(pair_count):
                    previous = previous_group[index]
                    current = unmatched_current[index]
                    assigned_previous.add(previous.object_id)
                    bound_by_temp_id[current.object_id] = replace(current, object_id=previous.object_id)
                unmatched_current = unmatched_current[pair_count:]
            for current in unmatched_current:
                bound_by_temp_id[current.object_id] = replace(current, object_id=self._new_track_id())
        return [bound_by_temp_id.get(obj.object_id, replace(obj, object_id=self._new_track_id())) for obj in objects]

    def _apply_role_labels(self, objects: list[ObjectState]) -> list[ObjectState]:
        labeled: list[ObjectState] = []
        for obj in objects:
            role_state = self.track_roles.get(obj.object_id)
            tags = set(obj.tags)
            if role_state is not None:
                if role_state.mover.confidence >= 0.65 and role_state.mover.balance > 0:
                    tags.add("repaired_mover")
                if role_state.goal.confidence >= 0.65 and role_state.goal.balance > 0:
                    tags.add("repaired_goal")
                if role_state.control.confidence >= 0.65 and role_state.control.balance > 0:
                    tags.add("repaired_control")
            labeled.append(replace(obj, tags=tuple(sorted(tags))))
        return labeled

    def _new_track_id(self) -> str:
        track_id = f"track_{self.next_track_index}"
        self.next_track_index += 1
        return track_id

    def _proposal_score(
        self,
        repair_type: str,
        features: tuple[float, ...],
        evidence: RepairEvidence | None,
    ) -> float:
        learned_score = self.proposal_model.score(repair_type, features)
        if evidence is None:
            return learned_score
        return (0.78 * learned_score) + (0.22 * evidence.confidence) + (0.03 * max(evidence.balance, 0))


def _match_signature(obj: ObjectState) -> MatchSignature:
    return (obj.color, tuple(sorted(tag for tag in obj.tags if not tag.startswith(_DYNAMIC_TAG_PREFIXES))))


def _group_by_match_signature(objects: tuple[ObjectState, ...] | list[ObjectState]) -> dict[MatchSignature, list[ObjectState]]:
    groups: dict[MatchSignature, list[ObjectState]] = defaultdict(list)
    for obj in objects:
        groups[_match_signature(obj)].append(obj)
    for group in groups.values():
        group.sort(key=_sort_key)
    return groups


def _sort_key(obj: ObjectState) -> tuple[float, float, int, str]:
    return (obj.centroid[0], obj.centroid[1], obj.area, obj.object_id)


def _sigmoid(value: float) -> float:
    clipped = max(min(value, 12.0), -12.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def _bbox_area(obj: ObjectState) -> float:
    height = (obj.bbox[2] - obj.bbox[0]) + 1
    width = (obj.bbox[3] - obj.bbox[1]) + 1
    return float(max(height * width, 1))


def _split_feature_vector(obj: ObjectState) -> tuple[float, ...] | None:
    split_objects = _split_object(obj)
    if len(split_objects) <= 1:
        return None
    height = (obj.bbox[2] - obj.bbox[0]) + 1
    width = (obj.bbox[3] - obj.bbox[1]) + 1
    fill_ratio = float(obj.area) / _bbox_area(obj)
    sizes = [split_obj.area for split_obj in split_objects]
    balance = float(min(sizes)) / float(max(max(sizes), 1))
    separation = _split_gap_score(split_objects)
    aspect = abs(height - width) / float(max(height, width, 1))
    return (
        1.0,
        math.tanh(float(obj.area) / 4.0),
        fill_ratio,
        balance,
        separation,
        aspect,
    )


def _merge_feature_vector(group: list[ObjectState]) -> tuple[float, ...]:
    merged = _merge_objects(group)
    pair_distances: list[float] = []
    for index, source in enumerate(group):
        for target in group[index + 1 :]:
            pair_distances.append(_object_cost(source, target))
    mean_distance = sum(pair_distances) / float(max(len(pair_distances), 1))
    fill_ratio = float(merged.area) / _bbox_area(merged)
    return (
        1.0,
        math.tanh(float(len(group) - 1)),
        fill_ratio,
        1.0 / float(mean_distance + 1.0),
        math.tanh(float(merged.area) / 6.0),
        _cluster_compactness(group),
    )


def _rebind_feature_vector(
    before_group: list[ObjectState],
    after_group: list[ObjectState],
    sorted_cost: float,
    nearest_cost: float,
) -> tuple[float, ...]:
    gap = max(sorted_cost - nearest_cost, 0.0)
    return (
        1.0,
        math.tanh(float(min(len(before_group), len(after_group))) / 3.0),
        math.tanh(gap),
        1.0 / float(nearest_cost + 1.0),
        1.0 / float(sorted_cost + 1.0),
    )


def _split_gap_score(split_objects: list[ObjectState]) -> float:
    if len(split_objects) < 2:
        return 0.0
    return min(
        abs(source.centroid[0] - target.centroid[0]) + abs(source.centroid[1] - target.centroid[1])
        for index, source in enumerate(split_objects)
        for target in split_objects[index + 1 :]
    ) / 4.0


def _cluster_compactness(group: list[ObjectState]) -> float:
    if len(group) <= 1:
        return 1.0
    mean_y = sum(obj.centroid[0] for obj in group) / float(len(group))
    mean_x = sum(obj.centroid[1] for obj in group) / float(len(group))
    dispersion = sum(abs(obj.centroid[0] - mean_y) + abs(obj.centroid[1] - mean_x) for obj in group) / float(len(group))
    return 1.0 / float(dispersion + 1.0)


def _object_cost(source: ObjectState, target: ObjectState) -> float:
    return (
        abs(source.centroid[0] - target.centroid[0])
        + abs(source.centroid[1] - target.centroid[1])
        + (0.5 * abs(source.area - target.area))
    )


def _ordered_group_cost(before_group: list[ObjectState], after_group: list[ObjectState]) -> float:
    pair_count = min(len(before_group), len(after_group))
    if pair_count <= 0:
        return 0.0
    return sum(_object_cost(before_group[index], after_group[index]) for index in range(pair_count)) / float(pair_count)


def _nearest_group_cost(before_group: list[ObjectState], after_group: list[ObjectState]) -> float:
    remaining = list(after_group)
    costs: list[float] = []
    for before_obj in before_group:
        if not remaining:
            break
        best_index = min(range(len(remaining)), key=lambda index: _object_cost(before_obj, remaining[index]))
        costs.append(_object_cost(before_obj, remaining[best_index]))
        remaining.pop(best_index)
    if not costs:
        return 0.0
    return sum(costs) / float(len(costs))


def _merge_candidate(source: ObjectState, target: ObjectState) -> bool:
    for source_cell in source.cells:
        for target_cell in target.cells:
            if abs(source_cell[0] - target_cell[0]) + abs(source_cell[1] - target_cell[1]) <= 2:
                return True
    return False


def _merge_objects(objects: list[ObjectState]) -> ObjectState:
    cells = tuple(sorted({cell for obj in objects for cell in obj.cells}))
    ys = [cell[0] for cell in cells]
    xs = [cell[1] for cell in cells]
    tags = tuple(sorted({tag for obj in objects for tag in obj.tags if tag not in _ROLE_TAGS}))
    return ObjectState(
        object_id=objects[0].object_id,
        color=objects[0].color,
        cells=cells,
        bbox=(min(ys), min(xs), max(ys), max(xs)),
        centroid=(float(sum(ys)) / len(ys), float(sum(xs)) / len(xs)),
        area=len(cells),
        tags=tags,
    )


def _split_object(obj: ObjectState) -> list[ObjectState]:
    vertical = _axis_split(obj, axis="x")
    horizontal = _axis_split(obj, axis="y")
    candidates = [candidate for candidate in (vertical, horizontal) if len(candidate) > 1]
    if not candidates:
        return [obj]
    best = max(
        candidates,
        key=lambda groups: min(len(group) for group in groups) - abs(len(groups[0]) - len(groups[1])),
    )
    split_objects: list[ObjectState] = []
    for index, cells in enumerate(best):
        ys = [cell[0] for cell in cells]
        xs = [cell[1] for cell in cells]
        split_objects.append(
            ObjectState(
                object_id=f"{obj.object_id}_split_{index}",
                color=obj.color,
                cells=tuple(sorted(cells)),
                bbox=(min(ys), min(xs), max(ys), max(xs)),
                centroid=(float(sum(ys)) / len(ys), float(sum(xs)) / len(xs)),
                area=len(cells),
                tags=tuple(sorted(tag for tag in obj.tags if tag not in _ROLE_TAGS)),
            )
        )
    return split_objects


def _axis_split(obj: ObjectState, *, axis: str) -> list[list[tuple[int, int]]]:
    coordinate_index = 1 if axis == "x" else 0
    values = sorted({cell[coordinate_index] for cell in obj.cells})
    best_groups: list[list[tuple[int, int]]] = []
    best_score = float("-inf")
    for left_value, right_value in zip(values, values[1:], strict=False):
        if right_value - left_value != 1:
            continue
        groups = [
            [cell for cell in obj.cells if cell[coordinate_index] <= left_value],
            [cell for cell in obj.cells if cell[coordinate_index] >= right_value],
        ]
        if not groups[0] or not groups[1]:
            continue
        score = min(len(groups[0]), len(groups[1])) - abs(len(groups[0]) - len(groups[1]))
        if score > best_score:
            best_score = score
            best_groups = groups
    return best_groups


def _state_delta(before: StructuredState, after: StructuredState) -> float:
    before_vector = before.transition_vector()
    after_vector = after.transition_vector()
    return sum(abs(float(before_value) - float(after_value)) for before_value, after_value in zip(before_vector, after_vector))


def _goal_activated(before: StructuredState, after: StructuredState) -> bool:
    before_active = before.flags_dict().get("goal_active") == "1" or any(
        "target" in obj.tags and "active" in obj.tags for obj in before.objects
    )
    after_active = after.flags_dict().get("goal_active") == "1" or any(
        "target" in obj.tags and "active" in obj.tags for obj in after.objects
    )
    return after_active and not before_active


def _object_motion(before: ObjectState, after: ObjectState) -> float:
    return abs(before.centroid[0] - after.centroid[0]) + abs(before.centroid[1] - after.centroid[1])


def _positive_reward_targets(before: StructuredState, after: StructuredState) -> set[str]:
    targets: set[str] = set()
    before_ids = {obj.object_id for obj in before.objects}
    after_ids = {obj.object_id for obj in after.objects}
    for obj in after.objects:
        if "target" in obj.tags or "repaired_goal" in obj.tags:
            targets.add(obj.object_id)
    for object_id in before_ids - after_ids:
        targets.add(object_id)
    return targets


def _interaction_target_object(
    state: StructuredState,
    action: ActionName,
    context,
) -> ObjectState | None:
    schema = build_action_schema(action, context)
    if schema.action_type == "interact":
        delta = _interaction_delta(schema.direction)
        if delta is None:
            return None
        agent_cells = [cell for obj in state.objects if "agent" in obj.tags for cell in obj.cells]
        if not agent_cells:
            return None
        target_cells = {(cell[0] + delta[0], cell[1] + delta[1]) for cell in agent_cells}
        for obj in state.objects:
            if "agent" in obj.tags:
                continue
            if any(cell in target_cells for cell in obj.cells):
                return obj
        return None
    if schema.action_type == "click" and schema.click is not None:
        click_x, click_y = schema.click
        for obj in state.objects:
            if (click_y, click_x) in obj.cells:
                return obj
    return None


def _interaction_delta(direction: str | None) -> tuple[int, int] | None:
    if direction == "up":
        return (-1, 0)
    if direction == "down":
        return (1, 0)
    if direction == "left":
        return (0, -1)
    if direction == "right":
        return (0, 1)
    return None


def _build_relations(objects: tuple[ObjectState, ...]) -> list[Relation]:
    relations: list[Relation] = []
    for source in objects:
        for target in objects:
            if source.object_id == target.object_id:
                continue
            dy = target.centroid[0] - source.centroid[0]
            dx = target.centroid[1] - source.centroid[1]
            manhattan = abs(dy) + abs(dx)
            if manhattan <= 3.0:
                relations.append(
                    Relation(
                        relation_type="near",
                        source_id=source.object_id,
                        target_id=target.object_id,
                        value=float(manhattan),
                    )
                )
            if dy < 0:
                relations.append(
                    Relation(
                        relation_type="above",
                        source_id=source.object_id,
                        target_id=target.object_id,
                        value=float(abs(dy)),
                    )
                )
            if dx < 0:
                relations.append(
                    Relation(
                        relation_type="left_of",
                        source_id=source.object_id,
                        target_id=target.object_id,
                        value=float(abs(dx)),
                    )
                )
    return relations
