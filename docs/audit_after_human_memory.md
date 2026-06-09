# Independent Audit Proof

Terminal outcome: **AUDIT PROVEN**

## Hashes

| Path | Before SHA256 | After SHA256 | Unchanged |
| --- | --- | --- | --- |
| `frozen/recurrent_latent_fast.pt` | `D36D59ED56A5BF4DC79835CB04D8B10F46E59FB00B2FE95DBF5AED30D1DBEFBD` | `D36D59ED56A5BF4DC79835CB04D8B10F46E59FB00B2FE95DBF5AED30D1DBEFBD` | True |
| `frozen/manifest.json` | `5A29A083287FDA14194392CE08CE24EDB56E21061285FFF2CCCB5E637807CA89` | `5A29A083287FDA14194392CE08CE24EDB56E21061285FFF2CCCB5E637807CA89` | True |
| `src/model.py` | `6C7D4FA2E8811DB76F12AF36DA20E56195F25F50DA349EA5335B26A8C90FDE1B` | `6C7D4FA2E8811DB76F12AF36DA20E56195F25F50DA349EA5335B26A8C90FDE1B` | True |
| `src/env.py` | `6EACA0F191E1849DF88C402500339F3E6B1C77FF6F0BB27DDF606FEC543FD04E` | `6EACA0F191E1849DF88C402500339F3E6B1C77FF6F0BB27DDF606FEC543FD04E` | True |
| `src/run_unbroken.py` | `004C239D211EF80CCA09637562908BD1C1FE716B0175810E0FD65FEA90CBEF89` | `004C239D211EF80CCA09637562908BD1C1FE716B0175810E0FD65FEA90CBEF89` | True |
| `src/persistent_memory.py` | `C24B4367437E61C1C19E8120C875276157EC729F10006E6ADBD69D016186BFC0` | `C24B4367437E61C1C19E8120C875276157EC729F10006E6ADBD69D016186BFC0` | True |
| `src/living_eval.py` | `134AC234A69356866B02212BAE8784BE4F58A2A141044FD46AC2CD1D25F97E53` | `134AC234A69356866B02212BAE8784BE4F58A2A141044FD46AC2CD1D25F97E53` | True |
| `src/curriculum.py` | `F94DBF04C81F897486ADFE6866569A696AF6562D7342304E35E33999EA8121CF` | `F94DBF04C81F897486ADFE6866569A696AF6562D7342304E35E33999EA8121CF` | True |
| `src/continual_learning.py` | `B56C723D9A8C6FBB71B5450F79D4897C99816F083422F165BD0973378CFEA3B1` | `B56C723D9A8C6FBB71B5450F79D4897C99816F083422F165BD0973378CFEA3B1` | True |
| `src/retention_eval.py` | `64121E109B1BC6827E38C10896C4136B1D27D384718F23821A30A0D1DC66A2A7` | `64121E109B1BC6827E38C10896C4136B1D27D384718F23821A30A0D1DC66A2A7` | True |
| `src/human_memory.py` | `A0AE42F8A9B12902AA6C6ADCFEA05BE8D10C0767FAB037147948148638C94713` | `A0AE42F8A9B12902AA6C6ADCFEA05BE8D10C0767FAB037147948148638C94713` | True |
| `src/memory_replay.py` | `5CF8DD8D8AD98CF2DE9031849C0EBCAF5F6835AEFA10EAB4432C30B9AE325F9D` | `5CF8DD8D8AD98CF2DE9031849C0EBCAF5F6835AEFA10EAB4432C30B9AE325F9D` | True |
| `src/memory_eval.py` | `D9C895D602F4CB7E09B1DC451342785EE35A636EF64592B05AC7B108DF28DD2A` | `D9C895D602F4CB7E09B1DC451342785EE35A636EF64592B05AC7B108DF28DD2A` | True |
| `src/train.py` | `00DEF2BD88364AA92663A40FBBE79FBD052DCA3DA876016A08473FF628306255` | `00DEF2BD88364AA92663A40FBBE79FBD052DCA3DA876016A08473FF628306255` | True |
| `src/evaluate.py` | `AF6403993DB80AA099E5CFAEB4B8BD68D0D07F0C5FF54C5339633CB7908E4E65` | `AF6403993DB80AA099E5CFAEB4B8BD68D0D07F0C5FF54C5339633CB7908E4E65` | True |
| `src/metrics.py` | `E06EFB02345A3568F7249BC3DAFC0E277421E873DF811F5BB49876E90D2801F4` | `E06EFB02345A3568F7249BC3DAFC0E277421E873DF811F5BB49876E90D2801F4` | True |
| `README.md` | `1DA55D70DCC20AB0C7C923D8631B6C1EA5384F207FF2119FE3AE665A0E035D33` | `1DA55D70DCC20AB0C7C923D8631B6C1EA5384F207FF2119FE3AE665A0E035D33` | True |
| `docs/living_system_report.json` | `2C859A655051667D2E37550B3B81E825BAF120B30DC8028354CF9C6FC8A38BFE` | `2C859A655051667D2E37550B3B81E825BAF120B30DC8028354CF9C6FC8A38BFE` | True |
| `docs/evidence_dossier.json` | `8021CC9CFE35128C76A8D1772C17B4621D92425FAA7C6F7E8D4A9E26161D3D06` | `8021CC9CFE35128C76A8D1772C17B4621D92425FAA7C6F7E8D4A9E26161D3D06` | True |

## Claim Proof Table

| Claim | Status | Evidence |
| --- | --- | --- |
| Frozen checkpoint and model match manifest before and after audit | PASS | SHA256 and size checks over frozen/recurrent_latent_fast.pt and src/model.py. |
| No external or pretrained model/API path is present | PASS | AST import scan, text pattern scan, weight-file scan, and runtime prompt-loop scan. |
| Living-system report metrics reproduce materially | PASS | Reran src.living_eval against the frozen checkpoint without overwriting the existing report. |
| Durable tensor memory survives restart and corrupted memory degrades | PASS | Saved and reloaded latent/private tensor state, then compared against sample-shuffled corrupt memory. |
| Idle low-input continuation preserves memory without repetition collapse | PASS | Reran idle eval and recorded one idle trajectory. |
| Private internal tokens are generated and causally consumed | PASS | Generated private-token stream plus zero/random-private perturbation shifts. |
| Richer body/world dynamics have consequences | PASS | Failed movement, rest, forage, and hazard probes. |
| Curriculum growth changes weights or persistent memory and retains old task behavior | PASS | Compared pre/post curriculum state, concept recall accuracy, and old-task retention. |
| Hidden targets and scoring data do not affect outputs | PASS | Randomized all target keys and added a hidden canary key while keeping observations fixed. |
| Latent stream is tensor-to-tensor, not generated public text | PASS | Source snippets show z is carried through model.step and private token, not language output, is fed as the private channel. |

## Metric Reproduction

Existing living report verdict reproduced: `True`.
Material metric differences over tolerance `0.02`: `0`.

## Anti-Leakage Probes

```json
{
  "scores": {
    "normal_generated_private": {
      "action_accuracy": 0.9679129464285714,
      "delayed_memory_accuracy": 0.9973958333333334,
      "object_color_accuracy": 0.9984809027777778,
      "object_pos_accuracy": 1.0,
      "object_permanence_accuracy": 0.9992404513888888,
      "provenance_accuracy": 0.9215262532234192,
      "grounded_language_accuracy": 0.9864583333333333,
      "self_world_continuity_accuracy": 1.0,
      "core_mean": 0.9838768400606654
    },
    "zero_z": {
      "action_accuracy": 0.48890904017857145,
      "delayed_memory_accuracy": 0.20963541666666666,
      "object_color_accuracy": 0.23531539351851852,
      "object_pos_accuracy": 0.2183159722222222,
      "object_permanence_accuracy": 0.22681568287037035,
      "provenance_accuracy": 0.6763392686843872,
      "grounded_language_accuracy": 0.28072916666666664,
      "self_world_continuity_accuracy": 0.19921875,
      "core_mean": 0.3169098363509254
    },
    "shuffled_z": {
      "action_accuracy": 0.36063058035714285,
      "delayed_memory_accuracy": 0.2209201388888889,
      "object_color_accuracy": 0.23625578703703703,
      "object_pos_accuracy": 0.1749855324074074,
      "object_permanence_accuracy": 0.2056206597222222,
      "provenance_accuracy": 0.9205496907234192,
      "grounded_language_accuracy": 0.2296875,
      "self_world_continuity_accuracy": 0.19921875,
      "core_mean": 0.3184835798920147
    },
    "no_language": {
      "action_accuracy": 0.4144810267857143,
      "delayed_memory_accuracy": 0.3072916666666667,
      "object_color_accuracy": 0.32740162037037035,
      "object_pos_accuracy": 0.33817997685185186,
      "object_permanence_accuracy": 0.3327907986111111,
      "provenance_accuracy": 0.3249860405921936,
      "grounded_language_accuracy": 0.0,
      "self_world_continuity_accuracy": 0.15625,
      "core_mean": 0.2751726412347385
    },
    "no_private_token": {
      "action_accuracy": 0.5537806919642857,
      "delayed_memory_accuracy": 0.3368055555555556,
      "object_color_accuracy": 0.4131944444444444,
      "object_pos_accuracy": 0.32125289351851855,
      "object_permanence_accuracy": 0.3672236689814815,
      "provenance_accuracy": 0.91259765625,
      "grounded_language_accuracy": 0.46484375,
      "self_world_continuity_accuracy": 0.9778645833333334,
      "core_mean": 0.5434454055059523
    },
    "random_private_token": {
      "action_accuracy": 0.5712890625,
      "delayed_memory_accuracy": 0.26171875,
      "object_color_accuracy": 0.36617476851851855,
      "object_pos_accuracy": 0.32544849537037035,
      "object_permanence_accuracy": 0.3458116319444444,
      "provenance_accuracy": 0.9102259874343872,
      "grounded_language_accuracy": 0.43385416666666665,
      "self_world_continuity_accuracy": 0.9153645833333334,
      "core_mean": 0.516235930720965
    }
  },
  "random_label_score": {
    "action_accuracy": 0.19168526785714285,
    "delayed_memory_accuracy": 0.2439236111111111,
    "object_color_accuracy": 0.24638310185185186,
    "object_pos_accuracy": 0.2016059027777778,
    "object_permanence_accuracy": 0.22399450231481483,
    "provenance_accuracy": 0.2466517835855484,
    "grounded_language_accuracy": 0.029947916666666668,
    "self_world_continuity_accuracy": 0.22786458333333334,
    "core_mean": 0.20150708368728085
  },
  "hidden_target_canary_max_abs_diff": 0.0,
  "no_private_action_shift_rate": 0.4760044515132904,
  "no_private_language_shift_rate": 0.3800223171710968,
  "checks": {
    "hidden_target_canary_no_effect": true,
    "random_labels_reduce_score": true,
    "zero_z_degrades_core": true,
    "shuffled_z_degrades_core": true,
    "no_language_degrades_grounding": true,
    "private_token_affects_outputs": true
  }
}
```

## Durable Memory

```json
{
  "restart_tick": 70,
  "normal_final_memory_accuracy": 0.9375,
  "memory_file_restart_final_memory_accuracy": 0.9375,
  "zero_reset_final_memory_accuracy": 0.25,
  "normal_final_object_pos_accuracy": 1.0,
  "memory_file_restart_final_object_pos_accuracy": 1.0,
  "zero_reset_final_object_pos_accuracy": 0.203125,
  "corrupt_memory_final_memory_accuracy": 0.2734375,
  "corrupt_memory_final_object_pos_accuracy": 0.1875,
  "checks": {
    "memory_file_good": true,
    "memory_beats_zero_reset": true,
    "corrupt_degrades_memory": true
  }
}
```

## Idle Trajectory

| Tick | Token | Visible | Private In | Generated Private | Action | Memory | Object Pos | Pass |
| ---: | --- | --- | ---: | ---: | --- | --- | --- | --- |
| 0 | OBSERVE_OBJECT | True | 0 | 1 | STAY | 0/0 | 2/2 | True |
| 3 | ASK_CURRENT_POS | True | 1 | 11 | LEFT | 0/0 | 2/2 | True |
| 8 | TOLD_GOAL | False | 1 | 1 | RIGHT | 0/0 | 2/2 | True |
| 16 | NONE | False | 7 | 13 | LEFT | 0/0 | 2/2 | True |
| 32 | NONE | False | 16 | 17 | RIGHT | 0/0 | 2/2 | True |
| 64 | NONE | False | 12 | 13 | RIGHT | 0/0 | 2/2 | True |
| 95 | NONE | False | 13 | 14 | LEFT | 0/0 | 2/2 | True |
| 111 | NONE | False | 17 | 12 | LEFT | 0/0 | 2/2 | True |

## Curriculum

```json
{
  "changed_parameter_tensors": 0,
  "parameter_l2_total": 0.0,
  "changes_weights": false,
  "changes_persistent_memory": true,
  "generated_private_curriculum_accuracy_before": 0.0,
  "generated_private_curriculum_accuracy_after": 1.0,
  "concept_memory_accuracy_before": 0.0,
  "concept_memory_accuracy_after": 1.0,
  "old_task_retention_before": {
    "action_accuracy": 0.9779296875,
    "delayed_memory_accuracy": 1.0,
    "object_color_accuracy": 1.0,
    "object_pos_accuracy": 1.0,
    "object_permanence_accuracy": 1.0,
    "provenance_accuracy": 0.999804675579071,
    "grounded_language_accuracy": 0.9859375,
    "self_world_continuity_accuracy": 1.0,
    "core_mean": 0.9954589828848839
  },
  "old_task_retention_after": {
    "action_accuracy": 0.9779296875,
    "delayed_memory_accuracy": 1.0,
    "object_color_accuracy": 1.0,
    "object_pos_accuracy": 1.0,
    "object_permanence_accuracy": 1.0,
    "provenance_accuracy": 0.999804675579071,
    "grounded_language_accuracy": 0.9859375,
    "self_world_continuity_accuracy": 1.0,
    "core_mean": 0.9954589828848839
  },
  "old_task_core_delta": 0.0,
  "old_task_submetric_floor_checks": {
    "action_accuracy": true,
    "delayed_memory_accuracy": true,
    "object_color_accuracy": true,
    "object_pos_accuracy": true,
    "object_permanence_accuracy": true,
    "provenance_accuracy": true,
    "grounded_language_accuracy": true,
    "self_world_continuity_accuracy": true
  },
  "old_task_retention_passes": true
}
```

## Required Source Snippets

### model_step_update
`src/model.py:60`

```python
60:     def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
61:         return torch.zeros(batch_size, self.config.hidden_dim, device=device)
62:
63:     def step(
64:         self, observation: Dict[str, torch.Tensor], z_prev: torch.Tensor
65:     ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
66:         sensory = observation["sensory"]
67:         lang_in = observation["lang_in"]
68:         private_in = observation.get("private_in")
69:         if private_in is None:
70:             private_in = torch.zeros_like(lang_in)
71:         sensor_features = self.sensor_encoder(sensory)
```

### runtime_z_loop
`src/run_unbroken.py:38`

```python
38:
39:     with torch.no_grad():
40:         for tick in range(max_ticks):
41:             observation = world.observation(device=device, private_in=int(private_token.item()))
42:             output, z = model.step(observation, z)
43:             action = int(output["action_logits"].argmax(dim=-1).item())
44:             language = int(output["language_logits"].argmax(dim=-1).item())
45:             private_token = output["private_logits"].argmax(dim=-1)
46:             latents.append(z.squeeze(0).detach().cpu())
47:             language_tokens.append(language)
48:             private_tokens.append(int(private_token.item()))
49:             if memory is not None:
50:                 memory.update(z, private_token, tick + 1)
51:             world.step(action)
52:             if log_every > 0 and (tick + 1) % log_every == 0:
53:                 print(
54:                     json.dumps(
55:                         {
56:                             "tick": tick + 1,
57:                             "z_norm": round(float(z.norm().item()), 6),
58:                             "action": action,
```

### memory_load_save
`src/persistent_memory.py:28`

```python
28:         )
29:
30:     @classmethod
31:     def load(
32:         cls,
33:         path: str | Path,
34:         hidden_dim: int,
35:         batch_size: int = 1,
36:         device: torch.device | str = "cpu",
37:     ) -> "PersistentMemoryState":
38:         path = Path(path)
39:         if not path.exists():
40:             return cls.fresh(hidden_dim, batch_size=batch_size, device=device)
41:         payload = torch.load(path, map_location=device)
42:         latent = payload["latent"].to(device).float()
43:         private_token = payload["private_token"].to(device).long()
44:         if latent.ndim == 1:
45:             latent = latent.view(1, -1)
46:         if private_token.ndim == 0:
47:             private_token = private_token.view(1)
48:         if latent.shape[-1] != hidden_dim:
49:             raise ValueError(f"memory latent width {latent.shape[-1]} does not match model hidden_dim {hidden_dim}")
50:         if latent.shape[0] != batch_size:
51:             latent = latent[:1].repeat(batch_size, 1)
52:         if private_token.shape[0] != batch_size:
53:             private_token = private_token[:1].repeat(batch_size)
54:         return cls(latent=latent, private_token=private_token, tick=int(payload.get("tick", 0)))
55:
56:     def update(self, latent: torch.Tensor, private_token: torch.Tensor, tick: int) -> None:
57:         self.latent = latent.detach().clone()
58:         self.private_token = private_token.detach().clone().long()
59:         self.tick = int(tick)
60:
61:     def save(self, path: str | Path) -> None:
62:         path = Path(path)
63:         path.parent.mkdir(parents=True, exist_ok=True)
64:         torch.save(
65:             {
66:                 "latent": self.latent.detach().cpu(),
67:                 "private_token": self.private_token.detach().cpu(),
68:                 "tick": self.tick,
69:                 "format": "persistent_differentiable_tensor_memory_v1",
70:             },
71:             path,
72:         )
```

### model_input_construction
`src/env.py:408`

```python
408:         self.damage = float(0.10 * torch.rand(1, generator=self.generator).item())
409:         self.resource = float(0.20 + 0.55 * torch.rand(1, generator=self.generator).item())
410:
411:     def observation(
412:         self,
413:         device: torch.device | str = "cpu",
414:         private_in: int = PRIVATE_NONE,
415:     ) -> Dict[str, torch.Tensor]:
416:         token = token_for_tick(self.local_tick)
417:         visible = self.target_color if self.local_tick < 4 else NUM_COLORS
418:         visible_pos = self.target_pos if self.local_tick < 4 else GRID_SIZE
419:         current_pos = torch.tensor([self.current_pos], dtype=torch.long)
420:         orientation = torch.tensor([self.orientation], dtype=torch.long)
421:         energy = torch.tensor([self.energy], dtype=torch.float32)
422:         fatigue = 1.0 - energy
423:         damage = torch.tensor([self.damage], dtype=torch.float32)
424:         resource = torch.tensor([self.resource], dtype=torch.float32)
425:         visible_color = torch.tensor([visible], dtype=torch.long)
426:         visible_object_pos = torch.tensor([visible_pos], dtype=torch.long)
427:         sensory = build_sensory(
428:             current_pos,
429:             orientation,
430:             energy,
431:             fatigue,
432:             damage,
433:             resource,
434:             visible_color,
435:             visible_object_pos,
436:         )
437:         return {
438:             "sensory": sensory.to(device),
439:             "lang_in": torch.tensor([token], dtype=torch.long, device=device),
440:             "private_in": torch.tensor([private_in], dtype=torch.long, device=device),
441:         }
442:
443:     def expected_action(self) -> int:
```

### target_input_separation
`src/env.py:350`

```python
350:         resource = torch.clamp(resource + forage.float() * 0.090 - rest.float() * 0.006 - 0.004, 0.0, 1.0)
351:
352:     private_in[:, 1:] = private[:, :-1]
353:
354:     batch = {
355:         "sensory": sensory,
356:         "lang_in": lang_in,
357:         "private_in": private_in,
358:         "action_target": action_target,
359:         "language_target": language,
360:         "private_target": private,
361:         "provenance_target": provenance,
362:         "world_color_target": world_color,
363:         "world_pos_target": world_pos,
364:         "memory_color_target": memory_color,
365:         "self_start_target": self_start,
366:         "action_mask": action_mask,
367:         "delayed_memory_mask": delayed_memory_mask,
368:         "object_mask": object_mask,
369:         "grounded_language_mask": grounded_language_mask,
370:         "self_mask": self_mask,
371:     }
372:     return {key: value.to(device) for key, value in batch.items()}
373:
374:
375: class TinyWorldRuntime:
376:     def __init__(self, seed: int = 0, episode_len: int = 80) -> None:
```

### private_token_generation_use
`src/living_eval.py:58`

```python
58:     outputs: List[Dict[str, torch.Tensor]] = []
59:     latents: List[torch.Tensor] = []
60:     private_tokens: List[torch.Tensor] = []
61:     for tick in range(start_tick, end_tick):
62:         output, z = model.step(
63:             {
64:                 "sensory": batch["sensory"][:, tick],
65:                 "lang_in": batch["lang_in"][:, tick],
66:                 "private_in": private_token,
67:             },
68:             z,
69:         )
70:         private_token = output["private_logits"].argmax(dim=-1)
71:         outputs.append(output)
72:         latents.append(z)
73:         private_tokens.append(private_token)
74:     stacked = {key: torch.stack([item[key] for item in outputs], dim=1) for key in outputs[0]}
75:     stacked["latents"] = torch.stack(latents, dim=1)
76:     stacked["generated_private"] = torch.stack(private_tokens, dim=1)
77:     return stacked
78:
79:
80: def durable_restart_eval(
```

### curriculum_update
`src/continual_learning.py:39`

```python
39:             concept_values=payload["concept_values"].to(device).long(),
40:         )
41:
42:     def clone(self) -> "PersistentConceptMemory":
43:         return PersistentConceptMemory(self.concept_ids.clone(), self.concept_values.clone())
44:
45:     def learn(self, concept_id: int, value: int | None = None) -> None:
46:         cid = torch.tensor([int(concept_id)], dtype=torch.long, device=self.concept_ids.device)
47:         val = torch.tensor([concept_value(concept_id) if value is None else int(value)], dtype=torch.long, device=self.concept_values.device)
48:         if self.concept_ids.numel() == 0:
49:             self.concept_ids = cid
50:             self.concept_values = val
51:             return
52:         exists = self.concept_ids == int(concept_id)
53:         if bool(exists.any().item()):
54:             self.concept_values[exists] = val[0]
55:         else:
56:             self.concept_ids = torch.cat([self.concept_ids, cid], dim=0)
57:             self.concept_values = torch.cat([self.concept_values, val], dim=0)
58:
59:     def predict(self, concept_ids: torch.Tensor) -> torch.Tensor:
60:         ids = concept_ids.to(self.concept_ids.device).long()
61:         pred = torch.full_like(ids, UNKNOWN_CONCEPT_VALUE)
62:         for concept_id, value in zip(self.concept_ids.tolist(), self.concept_values.tolist()):
63:             pred = torch.where(ids == int(concept_id), torch.full_like(pred, int(value)), pred)
```

### metric_computation
`src/living_eval.py:280`

```python
280:
281:
282: def living_verdict(report: Dict[str, object]) -> Dict[str, object]:
283:     restart = report["durable_restart"]  # type: ignore[assignment]
284:     idle = report["idle_mode"]  # type: ignore[assignment]
285:     dynamics = report["richer_dynamics"]  # type: ignore[assignment]
286:     private = report["private_internal_language"]  # type: ignore[assignment]
287:     curriculum = report["curriculum_growth"]  # type: ignore[assignment]
288:     checks = {
289:         "memory_file_restart_memory_ge_0_85": restart["memory_file_restart_final_memory_accuracy"] >= 0.85,
290:         "memory_file_beats_zero_reset_by_0_40": (
291:             restart["memory_file_restart_final_memory_accuracy"] - restart["zero_reset_final_memory_accuracy"] >= 0.40
292:         ),
293:         "idle_public_repetition_lt_0_40": idle["public_language_repetition_ratio"] < 0.40,
294:         "idle_private_repetition_lt_0_40": idle["private_token_repetition_ratio"] < 0.40,
295:         "idle_preserves_goal_memory": idle["final_memory_accuracy"] >= 0.85 and idle["final_object_pos_accuracy"] >= 0.85,
296:         "idle_endogenous_goal_action_ge_0_70": idle["endogenous_goal_action_accuracy"] >= 0.70,
297:         "richer_dynamics_all_present": all(value > 0.0 for value in dynamics.values()),
298:         "private_tokens_generated_and_used": (
299:             private["generated_private_unique_count"] >= 3.0
300:             and (
301:                 private["private_channel_action_shift_rate"] > 0.0
302:                 or private["private_channel_language_shift_rate"] > 0.0
303:             )
304:         ),
305:         "curriculum_accuracy_after_ge_0_80": curriculum["accuracy_after"] >= 0.80,
306:         "curriculum_improves_by_0_30": curriculum["accuracy_delta"] >= 0.30,
307:     }
308:     return {"passes": all(checks.values()), "checks": checks}
309:
310:
```

### leakage_scan_logic
`audit/leakage_scan.py:124`

```python
124:
125:
126: def run_scan() -> dict[str, Any]:
127:     python_files = _iter_python_files()
128:     findings: list[dict[str, Any]] = []
129:     for path in python_files:
130:         findings.extend(_import_findings(path))
131:         findings.extend(_text_findings(path))
132:     findings.extend(_checkpoint_findings())
133:     findings.extend(_prompt_loop_findings())
134:     return {
135:         "passes": len(findings) == 0,
136:         "python_files_scanned": [str(path) for path in python_files],
137:         "findings": findings,
138:         "checks": {
139:             "no_external_or_pretrained_imports": not any(item["kind"] == "forbidden_import" for item in findings),
140:             "no_external_or_pretrained_text_patterns": not any(item["kind"] == "external_pattern" for item in findings),
141:             "no_unexpected_weight_files": not any(item["kind"] == "unexpected_weight_file" for item in findings),
142:             "runtime_not_prompt_loop": not any("prompt" in item["kind"] for item in findings),
143:         },
144:     }
```

## Limitations
- None found by this audit.
