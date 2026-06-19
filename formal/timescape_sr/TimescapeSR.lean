import Std

/-!
# TimescapeSR

A deliberately minimal, assumption-transparent formal core for a kinematic
"timescape + special relativity" model.

The model has four layers.

1. `MinkowskiCore` supplies a special-relativistic interval and its null cone.
2. `TimescapeField` supplies a regional clock-calibration field (a lapse).
3. `PositiveConformalAction` scales the interval without changing its zero set;
   a `CausalClassifier` can additionally certify preservation of timelike/null/
   spacelike classes.
4. `RegionalClockAtlas` glues regional clock standards by a cocycle, while
   `TransitiveTranslations` exposes the no-go theorem: a globally translation-
   invariant lapse on a transitive spacetime is constant.

The optional quantum layer is intentionally weak: a `PhaseLaw` is a monoid
homomorphism from accumulated proper time to phases.  It proves composition of
regional proper-time phases, but it does not claim a quantum-gravity dynamics.
-/

namespace TimescapeSR

universe uEvent uScalar uScale uClass uTranslation uValue
universe uDuration uPhase uRegion uCalibration

/-! ## 1. Special-relativistic interval data -/

structure MinkowskiCore (Event : Type uEvent) (Scalar : Type uScalar) where
  interval : Event → Event → Scalar
  zero : Scalar

namespace MinkowskiCore

variable {Event : Type uEvent} {Scalar : Type uScalar}

/-- Two events are null-separated when their Minkowski interval is zero. -/
def Null (M : MinkowskiCore Event Scalar) (p q : Event) : Prop :=
  M.interval p q = M.zero

end MinkowskiCore

variable {Event : Type uEvent} {Scalar : Type uScalar}
variable {Scale : Type uScale} {Class : Type uClass}

/-- A Lorentz frame change, represented extensionally by interval preservation. -/
structure LorentzMap (M : MinkowskiCore Event Scalar) where
  toEquiv : Event ≃ Event
  preserves_interval :
    ∀ p q, M.interval (toEquiv p) (toEquiv q) = M.interval p q

/-! ## 2. Positive conformal clock calibration -/

/--
A scale action suitable for a positive conformal factor.  `zero_iff` is the
algebraic content needed to say that scaling neither creates nor destroys null
vectors.  Concrete models can instantiate `act a s = a^2 * s` with `a > 0`.
-/
structure PositiveConformalAction
    (Scale : Type uScale) (Scalar : Type uScalar) (zero : Scalar) where
  act : Scale → Scalar → Scalar
  zero_iff : ∀ a s, act a s = zero ↔ s = zero

/-- A scalar regional lapse/clock-calibration field. -/
structure TimescapeField (Event : Type uEvent) (Scale : Type uScale) where
  lapse : Event → Scale

namespace TimescapeField

variable {M : MinkowskiCore Event Scalar}

/--
The passive Lorentz transformation law for a scalar field: transform by
pulling the old field back along the inverse frame map.
-/
def pushforward (T : TimescapeField Event Scale) (L : LorentzMap M) :
    TimescapeField Event Scale where
  lapse x := T.lapse (L.toEquiv.symm x)

@[simp] theorem pushforward_lapse_at
    (T : TimescapeField Event Scale) (L : LorentzMap M) (p : Event) :
    (T.pushforward L).lapse (L.toEquiv p) = T.lapse p := by
  simp [pushforward]

end TimescapeField

/--
The local timescape interval anchored at `p`:

`I_T(p,q) = Ω(p)^2 I_η(p,q)` in a conventional real-valued realization.
-/
def scaledInterval
    (M : MinkowskiCore Event Scalar)
    (A : PositiveConformalAction Scale Scalar M.zero)
    (T : TimescapeField Event Scale)
    (p q : Event) : Scalar :=
  A.act (T.lapse p) (M.interval p q)

/--
Passive Lorentz covariance of the conformally calibrated interval.  Both the
coordinates and the scalar lapse field are transformed.
-/
theorem scaledInterval_covariant
    (M : MinkowskiCore Event Scalar)
    (A : PositiveConformalAction Scale Scalar M.zero)
    (T : TimescapeField Event Scale)
    (L : LorentzMap M) (p q : Event) :
    scaledInterval M A (T.pushforward L) (L.toEquiv p) (L.toEquiv q) =
      scaledInterval M A T p q := by
  simp [scaledInterval, TimescapeField.pushforward, L.preserves_interval]

/-- A positive conformal timescape factor preserves the Minkowski null cone. -/
theorem null_iff_scaled_null
    (M : MinkowskiCore Event Scalar)
    (A : PositiveConformalAction Scale Scalar M.zero)
    (T : TimescapeField Event Scale)
    (p q : Event) :
    M.Null p q ↔ scaledInterval M A T p q = M.zero := by
  unfold MinkowskiCore.Null scaledInterval
  exact (A.zero_iff (T.lapse p) (M.interval p q)).symm

/--
An abstract causal classifier.  In a real model this can be the sign of the
Minkowski interval, so its values represent timelike, null, and spacelike.
-/
structure CausalClassifier
    (Scale : Type uScale) (Scalar : Type uScalar) (Class : Type uClass)
    (zero : Scalar) (A : PositiveConformalAction Scale Scalar zero) where
  classify : Scalar → Class
  scale_invariant : ∀ a s, classify (A.act a s) = classify s

/-- Positive conformal clock calibration preserves every certified causal class. -/
theorem causal_class_preserved
    (M : MinkowskiCore Event Scalar)
    (A : PositiveConformalAction Scale Scalar M.zero)
    (C : CausalClassifier Scale Scalar Class M.zero A)
    (T : TimescapeField Event Scale)
    (p q : Event) :
    C.classify (scaledInterval M A T p q) =
      C.classify (M.interval p q) := by
  exact C.scale_invariant (T.lapse p) (M.interval p q)

/--
A genuine symmetry of a *fixed* timescape background must preserve both the
Minkowski interval and the lapse field.  This is stronger than passive
covariance under a change of frame.
-/
structure TimescapeLorentzSymmetry
    (M : MinkowskiCore Event Scalar) (T : TimescapeField Event Scale)
    extends LorentzMap M where
  preserves_lapse : ∀ p, T.lapse (toEquiv p) = T.lapse p

/-- A symmetry of the pair `(η, Ω)` preserves the calibrated interval. -/
theorem scaledInterval_invariant
    (M : MinkowskiCore Event Scalar)
    (A : PositiveConformalAction Scale Scalar M.zero)
    (T : TimescapeField Event Scale)
    (L : TimescapeLorentzSymmetry M T)
    (p q : Event) :
    scaledInterval M A T (L.toEquiv p) (L.toEquiv q) =
      scaledInterval M A T p q := by
  simp [scaledInterval, L.preserves_lapse, L.preserves_interval]

/-! ## 3. Regional clock atlas -/

/--
A groupoid-like atlas of regional clock standards.  `transition a b` converts
calibration data from region `a` to region `b`; `cocycle` makes the conversion
path-independent through one intermediate region.
-/
structure RegionalClockAtlas
    (Region : Type uRegion) (Calibration : Type uCalibration) where
  identity : Calibration
  compose : Calibration → Calibration → Calibration
  transition : Region → Region → Calibration
  identity_transition : ∀ r, transition r r = identity
  cocycle :
    ∀ a b c, compose (transition a b) (transition b c) = transition a c

/-- Passing through an intermediate region agrees with the direct calibration. -/
theorem regional_two_route_agreement
    (A : RegionalClockAtlas Region Calibration) (a b c : Region) :
    A.compose (A.transition a b) (A.transition b c) = A.transition a c := by
  exact A.cocycle a b c

/-- A round trip has trivial clock calibration (zero atlas holonomy). -/
theorem regional_round_trip_trivial
    (A : RegionalClockAtlas Region Calibration) (a b : Region) :
    A.compose (A.transition a b) (A.transition b a) = A.identity := by
  rw [A.cocycle a b a, A.identity_transition a]

/-- The complete kinematic package: local lapse plus a regional clock atlas. -/
structure RegionalTimescape
    (Event : Type uEvent) (Scale : Type uScale)
    (Region : Type uRegion) (Calibration : Type uCalibration) where
  field : TimescapeField Event Scale
  regionOf : Event → Region
  atlas : RegionalClockAtlas Region Calibration

/-! ## 4. Global-homogeneity no-go theorem -/

/-- A translation action that can carry any event to any other event. -/
structure TransitiveTranslations
    (Translation : Type uTranslation) (Event : Type uEvent) where
  act : Translation → Event → Event
  transitive : ∀ x y, ∃ a, act a x = y

/-- A field is invariant under every translation in `G`. -/
def TranslationInvariant
    (G : TransitiveTranslations Translation Event)
    (f : Event → Value) : Prop :=
  ∀ a x, f (G.act a x) = f x

/-- A field takes at least two different values. -/
def Nonconstant (f : Event → Value) : Prop :=
  ∃ x y, f x ≠ f y

/--
No-go theorem: on a transitive spacetime, every fully translation-invariant
clock-rate field is constant.
-/
theorem translation_invariant_constant
    (G : TransitiveTranslations Translation Event)
    (f : Event → Value)
    (h : TranslationInvariant G f) :
    ∀ x y, f x = f y := by
  intro x y
  obtain ⟨a, ha⟩ := G.transitive x y
  calc
    f x = f (G.act a x) := (h a x).symm
    _ = f y := congrArg f ha

/-- Contrapositive form: a nontrivial timescape breaks global translation symmetry. -/
theorem nonconstant_breaks_translation_invariance
    (G : TransitiveTranslations Translation Event)
    (f : Event → Value)
    (hNonconstant : Nonconstant f) :
    ¬ TranslationInvariant G f := by
  intro hInvariant
  obtain ⟨x, y, hxy⟩ := hNonconstant
  exact hxy (translation_invariant_constant G f hInvariant x y)

/-- The no-go result specialized to the lapse of a regional timescape model. -/
theorem nontrivial_timescape_not_globally_homogeneous
    (G : TransitiveTranslations Translation Event)
    (R : RegionalTimescape Event Scale Region Calibration)
    (hNonconstant : Nonconstant R.field.lapse) :
    ¬ TranslationInvariant G R.field.lapse := by
  exact nonconstant_breaks_translation_invariance G R.field.lapse hNonconstant

/-! ## 5. Calibrated proper time along worldlines -/

/--
An abstract additive law for SR proper durations plus regional calibration.
Concrete real-valued models use `append = (+)` and `calibrate Ω τ = Ω * τ`.
-/
structure ClockLaw (Scale : Type uScale) (Duration : Type uDuration) where
  zero : Duration
  append : Duration → Duration → Duration
  calibrate : Scale → Duration → Duration
  zero_append : ∀ d, append zero d = d
  append_zero : ∀ d, append d zero = d
  append_assoc :
    ∀ a b c, append (append a b) c = append a (append b c)
  calibrate_append :
    ∀ scale a b,
      calibrate scale (append a b) =
        append (calibrate scale a) (calibrate scale b)

/-- A short worldline segment, anchored where its regional lapse is sampled. -/
structure WorldlineSegment
    (Event : Type uEvent) (Duration : Type uDuration) where
  anchor : Event
  srProperTime : Duration

namespace WorldlineSegment

variable {M : MinkowskiCore Event Scalar}

/-- Lorentz-transform the anchor; SR proper time is carried as a scalar. -/
def map (L : LorentzMap M) (s : WorldlineSegment Event Duration) :
    WorldlineSegment Event Duration where
  anchor := L.toEquiv s.anchor
  srProperTime := s.srProperTime

end WorldlineSegment

/-- Regional clock reading on one SR worldline segment. -/
def segmentTime
    (C : ClockLaw Scale Duration)
    (T : TimescapeField Event Scale)
    (s : WorldlineSegment Event Duration) : Duration :=
  C.calibrate (T.lapse s.anchor) s.srProperTime

/-- Accumulated calibrated proper time on a piecewise worldline. -/
def pathTime
    (C : ClockLaw Scale Duration)
    (T : TimescapeField Event Scale) :
    List (WorldlineSegment Event Duration) → Duration
  | [] => C.zero
  | s :: rest => C.append (segmentTime C T s) (pathTime C T rest)

/-- Proper-time accumulation respects concatenation of piecewise worldlines. -/
theorem pathTime_append
    (C : ClockLaw Scale Duration)
    (T : TimescapeField Event Scale)
    (xs ys : List (WorldlineSegment Event Duration)) :
    pathTime C T (xs ++ ys) = C.append (pathTime C T xs) (pathTime C T ys) := by
  induction xs with
  | nil =>
      simp [pathTime, C.zero_append]
  | cons x xs ih =>
      simp [pathTime, ih, C.append_assoc]

/-- Two adjacent pieces at the same event can be compressed before calibration. -/
theorem same_anchor_clock_adds
    (C : ClockLaw Scale Duration)
    (T : TimescapeField Event Scale)
    (p : Event) (a b : Duration) :
    C.append
        (segmentTime C T ⟨p, a⟩)
        (segmentTime C T ⟨p, b⟩) =
      segmentTime C T ⟨p, C.append a b⟩ := by
  exact (C.calibrate_append (T.lapse p) a b).symm

/--
One-segment covariance: transforming the anchor and the scalar lapse field does
not alter the calibrated clock reading.
-/
theorem segmentTime_covariant
    {M : MinkowskiCore Event Scalar}
    (C : ClockLaw Scale Duration)
    (T : TimescapeField Event Scale)
    (L : LorentzMap M)
    (s : WorldlineSegment Event Duration) :
    segmentTime C (T.pushforward L) (s.map L) = segmentTime C T s := by
  simp [segmentTime, TimescapeField.pushforward, WorldlineSegment.map]

/-- Piecewise calibrated proper time is Lorentz-covariant. -/
theorem pathTime_covariant
    {M : MinkowskiCore Event Scalar}
    (C : ClockLaw Scale Duration)
    (T : TimescapeField Event Scale)
    (L : LorentzMap M)
    (xs : List (WorldlineSegment Event Duration)) :
    pathTime C (T.pushforward L) (xs.map (WorldlineSegment.map L)) =
      pathTime C T xs := by
  induction xs with
  | nil => rfl
  | cons s xs ih =>
      simp [pathTime, segmentTime_covariant, ih]

/-! ## 6. Optional quantum proper-time phase -/

/--
A quantum phase transport law over proper duration.  Physically, a standard
realization is `phase τ = exp(-i m c² τ / ℏ)`; only its composition law is used.
-/
structure PhaseLaw
    (Duration : Type uDuration) (Phase : Type uPhase)
    (zero : Duration) (append : Duration → Duration → Duration) where
  one : Phase
  mul : Phase → Phase → Phase
  phase : Duration → Phase
  phase_zero : phase zero = one
  phase_append : ∀ a b, phase (append a b) = mul (phase a) (phase b)

/-- Quantum phase assigned to a complete calibrated path. -/
def pathPhase
    (C : ClockLaw Scale Duration)
    (P : PhaseLaw Duration Phase C.zero C.append)
    (T : TimescapeField Event Scale)
    (xs : List (WorldlineSegment Event Duration)) : Phase :=
  P.phase (pathTime C T xs)

/-- The empty path has the identity phase. -/
theorem empty_path_phase
    (C : ClockLaw Scale Duration)
    (P : PhaseLaw Duration Phase C.zero C.append)
    (T : TimescapeField Event Scale) :
    pathPhase C P T [] = P.one := by
  simp [pathPhase, pathTime, P.phase_zero]

/-- Regional proper-time phases compose under path concatenation. -/
theorem pathPhase_append
    (C : ClockLaw Scale Duration)
    (P : PhaseLaw Duration Phase C.zero C.append)
    (T : TimescapeField Event Scale)
    (xs ys : List (WorldlineSegment Event Duration)) :
    pathPhase C P T (xs ++ ys) =
      P.mul (pathPhase C P T xs) (pathPhase C P T ys) := by
  unfold pathPhase
  rw [pathTime_append]
  exact P.phase_append (pathTime C T xs) (pathTime C T ys)

/-- The optional quantum phase inherits Lorentz covariance from proper time. -/
theorem pathPhase_covariant
    {M : MinkowskiCore Event Scalar}
    (C : ClockLaw Scale Duration)
    (P : PhaseLaw Duration Phase C.zero C.append)
    (T : TimescapeField Event Scale)
    (L : LorentzMap M)
    (xs : List (WorldlineSegment Event Duration)) :
    pathPhase C P (T.pushforward L) (xs.map (WorldlineSegment.map L)) =
      pathPhase C P T xs := by
  unfold pathPhase
  rw [pathTime_covariant]

end TimescapeSR
