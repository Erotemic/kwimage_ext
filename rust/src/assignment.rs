use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn value_error(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

/// Batched COCO-style greedy matching over a precomputed sparse candidate graph.
///
/// The candidate graph is CSR-like: prediction ``px`` owns the half-open slice
/// ``candidate_offsets[px]:candidate_offsets[px + 1]`` in ``candidate_txs`` and
/// ``candidate_ious``.  Each prediction's candidate slice must already be
/// ordered by descending overlap with the desired equal-IoU tie policy applied.
/// ``pred_order`` likewise supplies the stable score order after any maxDet or
/// prediction-ignore filtering.
///
/// The function intentionally performs matching only.  Category labels,
/// scores, weights, and confusion-row materialization stay in Python so this
/// kernel can remain small and reusable.
#[pyfunction]
#[pyo3(signature = (
    candidate_offsets,
    candidate_txs,
    candidate_ious,
    pred_order,
    thresholds,
    true_ignore_flags,
    annotation_ignore_flags,
    crowd_flags,
    inclusive=true
))]
pub fn coco_greedy_match<'py>(
    py: Python<'py>,
    candidate_offsets: PyReadonlyArray1<'py, i64>,
    candidate_txs: PyReadonlyArray1<'py, i64>,
    candidate_ious: PyReadonlyArray1<'py, f64>,
    pred_order: PyReadonlyArray1<'py, i64>,
    thresholds: PyReadonlyArray1<'py, f64>,
    true_ignore_flags: PyReadonlyArray1<'py, u8>,
    annotation_ignore_flags: PyReadonlyArray1<'py, u8>,
    crowd_flags: PyReadonlyArray1<'py, u8>,
    inclusive: bool,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let offsets = candidate_offsets.as_array();
    let txs = candidate_txs.as_array();
    let ious = candidate_ious.as_array();
    let pred_order = pred_order.as_array();
    let thresholds = thresholds.as_array();
    let true_ignore = true_ignore_flags.as_array();
    let annotation_ignore = annotation_ignore_flags.as_array();
    let crowd = crowd_flags.as_array();

    if offsets.is_empty() {
        return Err(value_error(
            "candidate_offsets must contain at least the initial zero offset",
        ));
    }
    let n_pred = offsets.len() - 1;
    let n_true = true_ignore.len();
    if annotation_ignore.len() != n_true || crowd.len() != n_true {
        return Err(value_error(format!(
            "truth flag arrays must have equal length: true_ignore={}, annotation_ignore={}, crowd={}",
            n_true,
            annotation_ignore.len(),
            crowd.len(),
        )));
    }
    if txs.len() != ious.len() {
        return Err(value_error(format!(
            "candidate_txs and candidate_ious length mismatch: {} != {}",
            txs.len(),
            ious.len(),
        )));
    }
    if offsets[0] != 0 {
        return Err(value_error("candidate_offsets[0] must be zero"));
    }
    let mut previous = 0i64;
    for (idx, &offset) in offsets.iter().enumerate() {
        if offset < previous {
            return Err(value_error(format!(
                "candidate_offsets must be monotonic; offset[{idx}]={offset} < {previous}",
            )));
        }
        if offset < 0 || offset as usize > txs.len() {
            return Err(value_error(format!(
                "candidate_offsets[{idx}]={offset} is outside flat candidate length {}",
                txs.len(),
            )));
        }
        previous = offset;
    }
    if offsets[n_pred] as usize != txs.len() {
        return Err(value_error(format!(
            "candidate_offsets[-1] must equal flat candidate length: {} != {}",
            offsets[n_pred],
            txs.len(),
        )));
    }

    let mut seen_pred = vec![false; n_pred];
    for &px_i64 in pred_order.iter() {
        if px_i64 < 0 || px_i64 as usize >= n_pred {
            return Err(value_error(format!(
                "pred_order contains out-of-range prediction index {px_i64} for n_pred={n_pred}",
            )));
        }
        let px = px_i64 as usize;
        if seen_pred[px] {
            return Err(value_error(format!(
                "pred_order contains duplicate prediction index {px}",
            )));
        }
        seen_pred[px] = true;
    }
    for &tx_i64 in txs.iter() {
        if tx_i64 < 0 || tx_i64 as usize >= n_true {
            return Err(value_error(format!(
                "candidate_txs contains out-of-range truth index {tx_i64} for n_true={n_true}",
            )));
        }
    }
    for &threshold in thresholds.iter() {
        if !threshold.is_finite() {
            return Err(value_error(format!(
                "thresholds must be finite; got {threshold}",
            )));
        }
    }

    let n_threshold = thresholds.len();
    let n_order = pred_order.len();
    let mut match_txs = Array2::<i64>::from_elem((n_threshold, n_order), -1);
    let mut match_ious = Array2::<f64>::from_elem((n_threshold, n_order), -1.0);
    let mut true_nmatches = Array2::<i64>::zeros((n_threshold, n_true));

    for ti in 0..n_threshold {
        let threshold = thresholds[ti];
        let mut true_available = vec![true; n_true];
        for tx in 0..n_true {
            if true_ignore[tx] != 0 {
                true_available[tx] = false;
            }
        }

        for order_pos in 0..n_order {
            let px = pred_order[order_pos] as usize;
            let start = offsets[px] as usize;
            let stop = offsets[px + 1] as usize;
            let mut chosen_tx: i64 = -1;
            let mut chosen_iou = -1.0f64;
            let mut ignored_fallback_tx: i64 = -1;
            let mut ignored_fallback_iou = -1.0f64;

            for cand_pos in start..stop {
                let cand_iou = ious[cand_pos];
                let passes = if inclusive {
                    cand_iou >= threshold
                } else {
                    cand_iou > threshold
                };
                if !passes {
                    // Candidate lists are pre-sorted by descending overlap.
                    break;
                }
                let tx = txs[cand_pos] as usize;
                if true_ignore[tx] != 0 {
                    continue;
                }
                if !true_available[tx] && crowd[tx] == 0 {
                    continue;
                }
                if annotation_ignore[tx] != 0 {
                    if ignored_fallback_tx < 0 {
                        ignored_fallback_tx = tx as i64;
                        ignored_fallback_iou = cand_iou;
                    }
                    continue;
                }
                chosen_tx = tx as i64;
                chosen_iou = cand_iou;
                break;
            }

            if chosen_tx < 0 && ignored_fallback_tx >= 0 {
                chosen_tx = ignored_fallback_tx;
                chosen_iou = ignored_fallback_iou;
            }

            if chosen_tx >= 0 {
                let tx = chosen_tx as usize;
                if crowd[tx] == 0 {
                    true_available[tx] = false;
                }
                true_nmatches[(ti, tx)] += 1;
                match_txs[(ti, order_pos)] = chosen_tx;
                match_ious[(ti, order_pos)] = chosen_iou;
            }
        }
    }

    Ok((
        match_txs.into_pyarray_bound(py),
        match_ious.into_pyarray_bound(py),
        true_nmatches.into_pyarray_bound(py),
    ))
}


/// Batched COCO-style greedy matching over an area x threshold grid.
///
/// This is the same semantic kernel as [`coco_greedy_match`], but
/// `annotation_ignore_flags` is a two-dimensional ``[area, truth]`` matrix.
/// Candidate geometry and prediction score order are shared across every area
/// slice, which lets downstream evaluators cross the Python/Rust boundary once
/// per image instead of once per image and area range.
#[pyfunction]
#[pyo3(signature = (
    candidate_offsets,
    candidate_txs,
    candidate_ious,
    pred_order,
    thresholds,
    true_ignore_flags,
    annotation_ignore_flags,
    crowd_flags,
    inclusive=true
))]
pub fn coco_greedy_match_grid<'py>(
    py: Python<'py>,
    candidate_offsets: PyReadonlyArray1<'py, i64>,
    candidate_txs: PyReadonlyArray1<'py, i64>,
    candidate_ious: PyReadonlyArray1<'py, f64>,
    pred_order: PyReadonlyArray1<'py, i64>,
    thresholds: PyReadonlyArray1<'py, f64>,
    true_ignore_flags: PyReadonlyArray1<'py, u8>,
    annotation_ignore_flags: PyReadonlyArray2<'py, u8>,
    crowd_flags: PyReadonlyArray1<'py, u8>,
    inclusive: bool,
) -> PyResult<(
    Bound<'py, PyArray3<i64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<i64>>,
)> {
    let offsets = candidate_offsets.as_array();
    let txs = candidate_txs.as_array();
    let ious = candidate_ious.as_array();
    let pred_order = pred_order.as_array();
    let thresholds = thresholds.as_array();
    let true_ignore = true_ignore_flags.as_array();
    let annotation_ignore = annotation_ignore_flags.as_array();
    let crowd = crowd_flags.as_array();

    if offsets.is_empty() {
        return Err(value_error(
            "candidate_offsets must contain at least the initial zero offset",
        ));
    }
    let n_pred = offsets.len() - 1;
    let n_true = true_ignore.len();
    let n_area = annotation_ignore.shape()[0];
    if annotation_ignore.shape()[1] != n_true || crowd.len() != n_true {
        return Err(value_error(format!(
            "truth flag arrays disagree: annotation_ignore={:?}, true_ignore={}, crowd={}",
            annotation_ignore.shape(),
            n_true,
            crowd.len(),
        )));
    }
    if txs.len() != ious.len() {
        return Err(value_error(format!(
            "candidate_txs and candidate_ious length mismatch: {} != {}",
            txs.len(),
            ious.len(),
        )));
    }
    if offsets[0] != 0 {
        return Err(value_error("candidate_offsets[0] must be zero"));
    }
    let mut previous = 0i64;
    for (idx, &offset) in offsets.iter().enumerate() {
        if offset < previous {
            return Err(value_error(format!(
                "candidate_offsets must be monotonic; offset[{idx}]={offset} < {previous}",
            )));
        }
        if offset < 0 || offset as usize > txs.len() {
            return Err(value_error(format!(
                "candidate_offsets[{idx}]={offset} is outside flat candidate length {}",
                txs.len(),
            )));
        }
        previous = offset;
    }
    if offsets[n_pred] as usize != txs.len() {
        return Err(value_error(format!(
            "candidate_offsets[-1] must equal flat candidate length: {} != {}",
            offsets[n_pred],
            txs.len(),
        )));
    }

    let mut seen_pred = vec![false; n_pred];
    for &px_i64 in pred_order.iter() {
        if px_i64 < 0 || px_i64 as usize >= n_pred {
            return Err(value_error(format!(
                "pred_order contains out-of-range prediction index {px_i64} for n_pred={n_pred}",
            )));
        }
        let px = px_i64 as usize;
        if seen_pred[px] {
            return Err(value_error(format!(
                "pred_order contains duplicate prediction index {px}",
            )));
        }
        seen_pred[px] = true;
    }
    for &tx_i64 in txs.iter() {
        if tx_i64 < 0 || tx_i64 as usize >= n_true {
            return Err(value_error(format!(
                "candidate_txs contains out-of-range truth index {tx_i64} for n_true={n_true}",
            )));
        }
    }
    for &threshold in thresholds.iter() {
        if !threshold.is_finite() {
            return Err(value_error(format!(
                "thresholds must be finite; got {threshold}",
            )));
        }
    }

    let n_threshold = thresholds.len();
    let n_order = pred_order.len();
    let mut match_txs = Array3::<i64>::from_elem((n_area, n_threshold, n_order), -1);
    let mut match_ious = Array3::<f64>::from_elem((n_area, n_threshold, n_order), -1.0);
    let mut true_nmatches = Array3::<i64>::zeros((n_area, n_threshold, n_true));

    for ai in 0..n_area {
        for ti in 0..n_threshold {
            let threshold = thresholds[ti];
            let mut true_available = vec![true; n_true];
            for tx in 0..n_true {
                if true_ignore[tx] != 0 {
                    true_available[tx] = false;
                }
            }

            for order_pos in 0..n_order {
                let px = pred_order[order_pos] as usize;
                let start = offsets[px] as usize;
                let stop = offsets[px + 1] as usize;
                let mut chosen_tx: i64 = -1;
                let mut chosen_iou = -1.0f64;
                let mut ignored_fallback_tx: i64 = -1;
                let mut ignored_fallback_iou = -1.0f64;

                for cand_pos in start..stop {
                    let cand_iou = ious[cand_pos];
                    let passes = if inclusive {
                        cand_iou >= threshold
                    } else {
                        cand_iou > threshold
                    };
                    if !passes {
                        break;
                    }
                    let tx = txs[cand_pos] as usize;
                    if true_ignore[tx] != 0 {
                        continue;
                    }
                    if !true_available[tx] && crowd[tx] == 0 {
                        continue;
                    }
                    if annotation_ignore[(ai, tx)] != 0 {
                        if ignored_fallback_tx < 0 {
                            ignored_fallback_tx = tx as i64;
                            ignored_fallback_iou = cand_iou;
                        }
                        continue;
                    }
                    chosen_tx = tx as i64;
                    chosen_iou = cand_iou;
                    break;
                }

                if chosen_tx < 0 && ignored_fallback_tx >= 0 {
                    chosen_tx = ignored_fallback_tx;
                    chosen_iou = ignored_fallback_iou;
                }

                if chosen_tx >= 0 {
                    let tx = chosen_tx as usize;
                    if crowd[tx] == 0 {
                        true_available[tx] = false;
                    }
                    true_nmatches[(ai, ti, tx)] += 1;
                    match_txs[(ai, ti, order_pos)] = chosen_tx;
                    match_ious[(ai, ti, order_pos)] = chosen_iou;
                }
            }
        }
    }

    Ok((
        match_txs.into_pyarray_bound(py),
        match_ious.into_pyarray_bound(py),
        true_nmatches.into_pyarray_bound(py),
    ))
}
