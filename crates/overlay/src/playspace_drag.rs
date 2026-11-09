//! Playspace drag corrections in STAGE coordinates (OpenXR Y is up).
//!
//! The located hand already includes the applied origin offset. Its distance
//! from the grab point is the remaining error, not a velocity to accumulate or
//! a raw hand position to low-pass: filtering that position would feed our own
//! origin changes back into the next sample. Apply only a fraction of the error
//! each frame, stopping inside a small dead zone. No pending motion survives
//! release, tracking loss, a snap back, or an axis change.

const DEAD_ZONE_M: f32 = 0.002;
const RESPONSE_SECS: f32 = 0.080;

pub fn correction(
    mut error: [f32; 3],
    vertical: bool,
    vertical_only: bool,
    stabilize: bool,
    dt: f32,
) -> [f32; 3] {
    if !error.iter().all(|v| v.is_finite()) {
        return [0.0; 3];
    }
    // Constrain before filtering, so ignored movement cannot influence height.
    if vertical_only {
        error[0] = 0.0;
        error[2] = 0.0;
    } else if !vertical {
        error[1] = 0.0;
    }
    let distance = error.iter().map(|v| v * v).sum::<f32>().sqrt();
    // Retain the existing outlier guard before smoothing, which could otherwise
    // shrink an invalid tracking jump enough to slip through the guard.
    if !distance.is_finite() || distance >= 10.0 {
        return [0.0; 3];
    }
    if !stabilize {
        return error;
    }
    if distance <= DEAD_ZONE_M || !dt.is_finite() || dt <= 0.0 {
        return [0.0; 3];
    }
    // A continuous dead zone avoids a step when deliberate motion begins.
    // Time-based gain gives the same response at different headset rates.
    // Cap a stalled frame's gain so resuming cannot produce a large correction.
    let gain = (1.0 - (-dt.min(0.025) / RESPONSE_SECS).exp()) * (distance - DEAD_ZONE_M) / distance;
    error.map(|v| v * gain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_only_blocks_both_horizontal_axes() {
        for vertical in [false, true] {
            for stabilize in [false, true] {
                let d = correction([0.8, -0.1, -0.6], vertical, true, stabilize, 1.0 / 90.0);
                assert_eq!(d[0], 0.0);
                assert_eq!(d[2], 0.0);
                assert!(d[1] < 0.0);
                assert_eq!(
                    d[1],
                    correction([0.0, -0.1, 0.0], vertical, true, stabilize, 1.0 / 90.0)[1]
                );
            }
        }
    }

    #[test]
    fn existing_modes_and_unsmoothed_motion_are_preserved() {
        assert_eq!(
            correction([0.1, 0.2, -0.3], true, false, false, 0.01),
            [0.1, 0.2, -0.3]
        );
        assert_eq!(
            correction([0.1, 0.2, -0.3], false, false, false, 0.01),
            [0.1, 0.0, -0.3]
        );
    }

    #[test]
    fn small_stationary_jitter_never_accumulates() {
        let mut offset = [0.0; 3];
        for i in 0..10_000 {
            let jitter = if i % 2 == 0 { 0.0015 } else { -0.0015 };
            let d = correction(
                [0.7, jitter - offset[1], -0.7],
                true,
                true,
                true,
                1.0 / 90.0,
            );
            for k in 0..3 {
                offset[k] += d[k];
            }
        }
        assert_eq!(offset, [0.0; 3]);
    }

    #[test]
    fn larger_tremor_is_attenuated_without_drift() {
        let mut offset = 0.0_f32;
        for i in 0..10_000 {
            let jitter = if i % 2 == 0 { 0.003 } else { -0.003 };
            offset += correction([0.0, jitter - offset, 0.0], true, true, true, 1.0 / 90.0)[1];
            assert!(offset.abs() < 0.0003, "offset {offset}");
        }
    }

    #[test]
    fn deliberate_motion_settles_at_different_refresh_rates() {
        for hz in [60, 72, 90, 120, 144] {
            // Simulate two frames of runtime latency between a correction and
            // the tracked pose reflecting it. This must settle without shaking.
            let mut history = [0.0; 3];
            let mut offset = 0.0_f32;
            for _ in 0..hz {
                let d = correction(
                    [0.0, 0.2 - history[0], 0.0],
                    true,
                    true,
                    true,
                    1.0 / hz as f32,
                );
                offset += d[1];
                history.rotate_left(1);
                history[2] = offset;
                assert!((0.0..0.203).contains(&offset), "{hz} Hz overshot: {offset}");
            }
            assert!((offset - 0.2).abs() < 0.003, "{hz} Hz: {offset}");
            // Releasing calls no correction; a new grab at the current hand
            // pose starts at zero error and inherits no old filter state.
            assert_eq!(
                correction([0.0; 3], true, true, true, 1.0 / hz as f32),
                [0.0; 3]
            );
        }
    }

    #[test]
    fn response_is_independent_of_refresh_rate() {
        for hz in [60, 72, 90, 120, 144] {
            let mut offset = 0.0;
            for _ in 0..hz {
                offset +=
                    correction([0.0, 0.2 - offset, 0.0], true, true, true, 0.25 / hz as f32)[1];
            }
            let expected = (0.2 - DEAD_ZONE_M) * (1.0 - (-0.25 / RESPONSE_SECS).exp());
            assert!((offset - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn invalid_samples_and_stalls_do_not_poison_the_offset() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                correction([value, 0.1, 0.0], true, true, true, 0.01),
                [0.0; 3]
            );
            assert_eq!(
                correction([0.0, 0.1, 0.0], true, true, true, value),
                [0.0; 3]
            );
        }
        assert_eq!(correction([0.0, 0.1, 0.0], true, true, true, 0.0), [0.0; 3]);
        for stabilize in [false, true] {
            assert_eq!(
                correction([0.0, 10.0, 0.0], true, true, stabilize, 0.01),
                [0.0; 3]
            );
            assert_eq!(
                correction([0.0, f32::MAX, 0.0], true, true, stabilize, 0.01),
                [0.0; 3]
            );
        }
        assert_eq!(
            correction([0.0, 0.1, 0.0], true, true, true, 1.0),
            correction([0.0, 0.1, 0.0], true, true, true, 0.025)
        );
    }
}
