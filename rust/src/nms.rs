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

    // Keep the existing general strided semantics, but shed ndarray row-view
    // construction and dynamic stride arithmetic for normal contiguous inputs.
    if boxes.is_standard_layout() && scores.is_standard_layout() {
        let boxes_data = boxes
            .as_slice()
            .expect("standard-layout boxes must expose a contiguous slice");
        let scores_data = scores
            .as_slice()
            .expect("standard-layout scores must expose a contiguous slice");

        let areas: Vec<f32> = boxes_data
            .chunks_exact(4)
            .map(|b| (b[2] - b[0] + bias) * (b[3] - b[1] + bias))
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            scores_data[b]
                .partial_cmp(&scores_data[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        // Vec<bool> is bit-packed. NMS reads this flag in its innermost loop,
        // so a byte-per-entry representation is cheaper and still tiny.
        let mut suppressed = vec![0u8; n];
        let mut keep = Vec::with_capacity(n);
        for oi in 0..order.len() {
            let i = order[oi];
            if suppressed[i] != 0 {
                continue;
            }
            keep.push(i);
            let ibase = i * 4;
            let ix1 = boxes_data[ibase];
            let iy1 = boxes_data[ibase + 1];
            let ix2 = boxes_data[ibase + 2];
            let iy2 = boxes_data[ibase + 3];
            let iarea = areas[i];

            for &j in &order[(oi + 1)..] {
                if suppressed[j] != 0 {
                    continue;
                }
                let jbase = j * 4;
                let xx1 = ix1.max(boxes_data[jbase]);
                let yy1 = iy1.max(boxes_data[jbase + 1]);
                let xx2 = ix2.min(boxes_data[jbase + 2]);
                let yy2 = iy2.min(boxes_data[jbase + 3]);
                let w = (xx2 - xx1 + bias).max(0.0);
                let h = (yy2 - yy1 + bias).max(0.0);
                let inter = w * h;
                let denom = iarea + areas[j] - inter;
                let overlap = if denom == 0.0 { 0.0 } else { inter / denom };
                if overlap > thresh {
                    suppressed[j] = 1;
                }
            }
        }
        return Ok(keep);
    }

    let areas: Vec<f32> = (0..n)
        .map(|i| {
            let b = boxes.row(i);
            (b[2] - b[0] + bias) * (b[3] - b[1] + bias)
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            // Match the historical CPU-NMS behavior for equal scores on the
            // supported NumPy stack: earlier input indices win ties. Make the
            // rule explicit here instead of inheriting an unstable sort detail.
            .then_with(|| a.cmp(&b))
    });
    let mut suppressed = vec![0u8; n];
    let mut keep = Vec::with_capacity(n);
    for oi in 0..order.len() {
        let i = order[oi];
        if suppressed[i] != 0 {
            continue;
        }
        keep.push(i);
        let ib = boxes.row(i);
        for oj in (oi + 1)..order.len() {
            let j = order[oj];
            if suppressed[j] != 0 {
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
                suppressed[j] = 1;
            }
        }
    }
    Ok(keep)
}

fn soft_nms_contiguous(
    boxes: &mut [f32],
    scores: &mut [f32],
    thresh: f32,
    overlap_thresh: f32,
    sigma: f32,
    bias: f32,
    method: u32,
) -> Vec<isize> {
    let mut active_n = scores.len();
    let mut inds: Vec<isize> = (0..active_n).map(|idx| idx as isize).collect();
    let mut i = 0usize;

    while i < active_n {
        let mut maxpos = i;
        let mut maxscore = scores[i];
        for pos in (i + 1)..active_n {
            if scores[pos] > maxscore {
                maxscore = scores[pos];
                maxpos = pos;
            }
        }
        if maxpos != i {
            let ibase = i * 4;
            let maxbase = maxpos * 4;
            for c in 0..4 {
                boxes.swap(ibase + c, maxbase + c);
            }
            scores.swap(i, maxpos);
            inds.swap(i, maxpos);
        }

        let ibase = i * 4;
        let tx1 = boxes[ibase];
        let ty1 = boxes[ibase + 1];
        let tx2 = boxes[ibase + 2];
        let ty2 = boxes[ibase + 3];
        let selected_area = (tx2 - tx1 + bias) * (ty2 - ty1 + bias);

        let mut pos = i + 1;
        while pos < active_n {
            let base = pos * 4;
            let x1 = boxes[base];
            let y1 = boxes[base + 1];
            let x2 = boxes[base + 2];
            let y2 = boxes[base + 3];

            let area = (x2 - x1 + bias) * (y2 - y1 + bias);
            let iw = tx2.min(x2) - tx1.max(x1) + bias;
            if iw > 0.0 {
                let ih = ty2.min(y2) - ty1.max(y1) + bias;
                if ih > 0.0 {
                    let inter = iw * ih;
                    let denom = selected_area + area - inter;
                    let ov = if denom == 0.0 { 0.0 } else { inter / denom };
                    let weight = match method {
                        1 => {
                            if ov > overlap_thresh { 1.0 - ov } else { 1.0 }
                        }
                        2 => (-(ov * ov) / sigma).exp(),
                        _ => {
                            if ov > overlap_thresh { 0.0 } else { 1.0 }
                        }
                    };
                    scores[pos] *= weight;
                }
            }

            if scores[pos] < thresh {
                let last = active_n - 1;
                if pos != last {
                    let last_base = last * 4;
                    for c in 0..4 {
                        boxes[base + c] = boxes[last_base + c];
                    }
                    scores[pos] = scores[last];
                    inds[pos] = inds[last];
                }
                active_n -= 1;
                continue;
            }
            pos += 1;
        }
        i += 1;
    }

    inds.truncate(active_n);
    inds
}

fn soft_nms_strided(
    boxes: &mut numpy::ndarray::ArrayViewMut2<'_, f32>,
    scores: &mut numpy::ndarray::ArrayViewMut1<'_, f32>,
    thresh: f32,
    overlap_thresh: f32,
    sigma: f32,
    bias: f32,
    method: u32,
) -> Vec<isize> {
    let mut active_n = boxes.nrows();
    let mut inds: Vec<isize> = (0..active_n).map(|idx| idx as isize).collect();
    let mut i = 0usize;

    while i < active_n {
        let mut maxpos = i;
        let mut maxscore = scores[i];
        for pos in (i + 1)..active_n {
            if scores[pos] > maxscore {
                maxscore = scores[pos];
                maxpos = pos;
            }
        }
        if maxpos != i {
            for c in 0..4 {
                let tmp = boxes[(i, c)];
                boxes[(i, c)] = boxes[(maxpos, c)];
                boxes[(maxpos, c)] = tmp;
            }
            scores.swap(i, maxpos);
            inds.swap(i, maxpos);
        }

        let tx1 = boxes[(i, 0)];
        let ty1 = boxes[(i, 1)];
        let tx2 = boxes[(i, 2)];
        let ty2 = boxes[(i, 3)];
        let selected_area = (tx2 - tx1 + bias) * (ty2 - ty1 + bias);

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
                    let inter = iw * ih;
                    let denom = selected_area + area - inter;
                    let ov = if denom == 0.0 { 0.0 } else { inter / denom };
                    let weight = match method {
                        1 => {
                            if ov > overlap_thresh { 1.0 - ov } else { 1.0 }
                        }
                        2 => (-(ov * ov) / sigma).exp(),
                        _ => {
                            if ov > overlap_thresh { 0.0 } else { 1.0 }
                        }
                    };
                    scores[pos] *= weight;
                }
            }

            if scores[pos] < thresh {
                let last = active_n - 1;
                if pos != last {
                    for c in 0..4 {
                        let last_value = boxes[(last, c)];
                        boxes[(pos, c)] = last_value;
                    }
                    scores[pos] = scores[last];
                    inds[pos] = inds[last];
                }
                active_n -= 1;
                continue;
            }
            pos += 1;
        }
        i += 1;
    }

    inds.truncate(active_n);
    inds
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

    let inds = if boxes.is_standard_layout() && score_view.is_standard_layout() {
        let boxes_data = boxes
            .as_slice_mut()
            .expect("standard-layout boxes must expose a contiguous slice");
        let score_data = score_view
            .as_slice_mut()
            .expect("standard-layout scores must expose a contiguous slice");
        soft_nms_contiguous(
            boxes_data,
            score_data,
            thresh,
            overlap_thresh,
            sigma,
            bias,
            method,
        )
    } else {
        soft_nms_strided(
            &mut boxes,
            &mut score_view,
            thresh,
            overlap_thresh,
            sigma,
            bias,
            method,
        )
    };

    Ok(inds.into_pyarray_bound(py))
}
