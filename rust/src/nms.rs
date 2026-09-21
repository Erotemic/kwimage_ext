use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;

fn check_ltrb_shape(shape: &[usize]) -> PyResult<()> {
    if shape.len() != 2 || shape[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("ltrb must have shape Nx4, got {:?}", shape)
        ));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (ltrb, scores, thresh, bias=0.0))]
pub fn cpu_nms(
    ltrb: PyReadonlyArray2<'_, f32>,
    scores: PyReadonlyArray1<'_, f32>,
    thresh: f32,
    bias: f32,
) -> PyResult<Vec<usize>> {
    let boxes = ltrb.as_array();
    let scores = scores.as_array();
    check_ltrb_shape(boxes.shape())?;
    if boxes.nrows() != scores.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "scores length must equal number of boxes"
        ));
    }
    let n = boxes.nrows();
    let areas: Vec<f32> = (0..n).map(|i| {
        let b = boxes.row(i);
        (b[2] - b[0] + bias) * (b[3] - b[1] + bias)
    }).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
            // Match the historical CPU-NMS behavior for equal scores on the
            // supported NumPy stack: earlier input indices win ties.  Make the
            // rule explicit here instead of inheriting an unstable sort detail.
            .then_with(|| a.cmp(&b))
    });
    let mut suppressed = vec![false; n];
    let mut keep = Vec::with_capacity(n);
    for oi in 0..order.len() {
        let i = order[oi];
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        let ib = boxes.row(i);
        for oj in (oi + 1)..order.len() {
            let j = order[oj];
            if suppressed[j] {
                continue;
            }
            let jb = boxes.row(j);
            let xx1 = ib[0].max(jb[0]);
            let yy1 = ib[1].max(jb[1]);
            let xx2 = ib[2].min(jb[2]);
            let yy2 = ib[3].min(jb[3]);
            let w = (xx2 - xx1 + bias).max(0.0);
            let h = (yy2 - yy1 + bias).max(0.0);
            let inter = w * h;
            let denom = areas[i] + areas[j] - inter;
            let overlap = if denom == 0.0 { 0.0 } else { inter / denom };
            if overlap > thresh {
                suppressed[j] = true;
            }
        }
    }
    Ok(keep)
}

/// Rust port of the historical Cython Soft-NMS implementation.
///
/// Like the legacy implementation, this mutates ``ltrb`` and ``scores`` in
/// place while returning the original indices that remain active.
#[pyfunction]
#[pyo3(signature = (
    ltrb,
    scores,
    thresh=0.001,
    overlap_thresh=0.3,
    sigma=0.5,
    bias=0.0,
    method=0
))]
pub fn soft_nms<'py>(
    py: Python<'py>,
    mut ltrb: PyReadwriteArray2<'py, f32>,
    mut scores: PyReadwriteArray1<'py, f32>,
    thresh: f32,
    overlap_thresh: f32,
    sigma: f32,
    bias: f32,
    method: u32,
) -> PyResult<Bound<'py, PyArray1<isize>>> {
    let mut boxes = ltrb.as_array_mut();
    let mut score_view = scores.as_array_mut();
    check_ltrb_shape(boxes.shape())?;
    if boxes.nrows() != score_view.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "scores length must equal number of boxes"
        ));
    }

    let mut active_n = boxes.nrows();
    let mut inds: Vec<isize> = (0..active_n).map(|idx| idx as isize).collect();
    let mut i = 0usize;

    while i < active_n {
        // Move the highest-scoring remaining detection into position i.
        let mut maxpos = i;
        let mut maxscore = score_view[i];
        for pos in (i + 1)..active_n {
            if score_view[pos] > maxscore {
                maxscore = score_view[pos];
                maxpos = pos;
            }
        }
        if maxpos != i {
            for c in 0..4 {
                let tmp = boxes[(i, c)];
                boxes[(i, c)] = boxes[(maxpos, c)];
                boxes[(maxpos, c)] = tmp;
            }
            score_view.swap(i, maxpos);
            inds.swap(i, maxpos);
        }

        let tx1 = boxes[(i, 0)];
        let ty1 = boxes[(i, 1)];
        let tx2 = boxes[(i, 2)];
        let ty2 = boxes[(i, 3)];

        let mut pos = i + 1;
        while pos < active_n {
            let x1 = boxes[(pos, 0)];
            let y1 = boxes[(pos, 1)];
            let x2 = boxes[(pos, 2)];
            let y2 = boxes[(pos, 3)];

            let area = (x2 - x1 + bias) * (y2 - y1 + bias);
            let iw = tx2.min(x2) - tx1.max(x1) + bias;
            if iw > 0.0 {
                let ih = ty2.min(y2) - ty1.max(y1) + bias;
                if ih > 0.0 {
                    let denom =
                        (tx2 - tx1 + bias) * (ty2 - ty1 + bias) + area - iw * ih;
                    let ov = if denom == 0.0 { 0.0 } else { iw * ih / denom };
                    let weight = match method {
                        1 => {
                            if ov > overlap_thresh { 1.0 - ov } else { 1.0 }
                        }
                        2 => (-(ov * ov) / sigma).exp(),
                        _ => {
                            if ov > overlap_thresh { 0.0 } else { 1.0 }
                        }
                    };
                    score_view[pos] *= weight;
                }
            }

            if score_view[pos] < thresh {
                let last = active_n - 1;
                if pos != last {
                    // The historical Cython implementation *copies* the last
                    // active entry over the discarded slot rather than swapping
                    // it.  The inactive tail is externally observable because
                    // ltrb/scores are mutated in place, so preserve that exact
                    // mutation behavior for compatibility.
                    for c in 0..4 {
                        let last_value = boxes[(last, c)];
                        boxes[(pos, c)] = last_value;
                    }
                    let last_score = score_view[last];
                    score_view[pos] = last_score;
                    let last_index = inds[last];
                    inds[pos] = last_index;
                }
                active_n -= 1;
                // Inspect the item copied into `pos` before advancing.
                continue;
            }
            pos += 1;
        }
        i += 1;
    }

    Ok(inds[..active_n].to_vec().into_pyarray_bound(py))
}
