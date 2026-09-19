use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

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
    if boxes.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("ltrb must have shape Nx4, got {:?}", boxes.shape())
        ));
    }
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
            .then_with(|| b.cmp(&a))
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
